import abc

import clip.simple_tokenizer
import p2p.ptp_utils as ptp_utils
import p2p.seq_aligner as seq_aligner
import torch
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict
import numpy as np
import clip
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW_RESOURCE = False
MY_TOKEN = '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'
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
#CLIP,_=clip.load('ViT-B/32',device)
tokenizer = clip.simple_tokenizer.SimpleTokenizer()

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)#[batch,1,H,w]
        mask = (mask[:1] + mask[1:]).float()#[batch-1,1,H,w]
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3,device=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place)
            else:
                #对于condition control
                # h = attn.shape[0]
                # attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place)
                #?
                attn = self.forward(attn, is_cross, place)
        self.cur_att_layer += 1
        #完整执行一步
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            #本地存储attention
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place: str):
        return attn
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return{"layer0_self":[],"layer1_self":[],"layer2_self":[],"layer3_self":[],"layer4_self":[],"layer5_self":[],"layer6_self":[],"layer7_self":[]
               ,"layer0_cross":[],"layer1_cross":[],"layer2_cross":[],"layer3_cross":[],"layer4_cross":[],"layer5_cross":[],"layer6_cross":[],"layer7_cross":[]}
        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place: str):
        key = f"{place}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        #     self.step_store[key].append(attn)
        #attn [batch,frame,77,8]
        self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        #?
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            #[batch,frame,77,8]
            h = attn.shape[0] // (self.batch_size)

            #attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            #attn[0]原prompt对应 attn[1:]replace后的attn
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                #alpha_words[1,1,77]
                alpha_words = self.cross_replace_alpha[self.cur_step].unsqueeze(-1)
                #replace_attention[batch,frame,77,8] attn_base[frame,77,8] attn_base[batch,frame,77,8] 
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            #attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],device=None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.device=device
        #cross_replace_alpha[steps,batch,1,1,77]
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        #注意力替换区间
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bpn->bhnw', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,device=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(self.device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, self.mapper,:].permute(1, 3, 0, 2)#[1,head,frame,77]
        attn_replace = attn_base_replace * self.alphas + att_replace.permute(0,3,1,2) * (1 - self.alphas)
        return attn_replace.permute(0,2,3,1)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,device=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,device)
        #[batch,77] alphas表示是否有对应词
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        #[batch,77]
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        #[batch,1,1,77]
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :].permute(0,1,3,2)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(self.device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    #qualizer[values,77]
    return equalizer

def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    ptp_utils.view_images(images)
    return images, x_t