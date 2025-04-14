import torch
from typing import Optional
from torch.nn.attention import SDPBackend

def sample_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None, vocab_size=8192):
    """
    Sample from the logits using top-k sampling.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    # logits: [batch_size, seq_len, vocab_size]
    if temperature == 0.0:
        idx_next = torch.argmax(logits[:, -1, :vocab_size], dim=-1, keepdim=True)
    else:
        probs = logits_to_probs(logits[:, -1, :vocab_size], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next

def multinomial_sample_one_no_sync(probs_sort, dtype=torch.int):
    """
    Multinomial sampling without a cuda synchronization.
    Source: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=dtype)

def logits_to_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample_top_p(logits, temperature, top_p, vocab_size=8192):
    probs = torch.softmax(logits[:, -1, :vocab_size] / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample_n_top_p(logits, temperature, top_p, vocab_size=8192):
    probs = torch.softmax(logits[:, :, :vocab_size] / temperature, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial_sample_one_no_sync(probs_sort, dtype=torch.int64)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample_n_top_k(logits, temperature: float = 1.0, top_k: Optional[int] = None, vocab_size=8192):
    if temperature == 0.0:
        # Modify for multiple logits (n items)
        idx_next = torch.argmax(logits[:, :, :vocab_size], dim=-1, keepdim=True)  # Use all n logits for top-k
        probs = None
    else:
        probs = logits_to_n_probs(logits[:, :, :vocab_size], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)

    return idx_next

def logits_to_n_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def decode_one_token(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """
    Decode a single token from the autoregressive model.
    """
    logits = model(input_ids=input_ids, position_ids=position_ids)
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)
    
def decode_some_token(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """
    Decode multi token from the autoregressive model.
    """
    logits = model(input_ids=input_ids, position_ids=position_ids)
    if top_p is not None:
        return sample_n_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_n_top_k(logits, temperature=temperature, top_k=top_k)
    
def decode_n_tokens(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_one_token_function=decode_one_token,
    pixnum: int = 336,
    actnum: int = 11,
    **kwargs,
):
    """
    Decode n tokens from the autoregressive model.
    Adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    new_tokens = [input_ids]
    pos_ = position_ids
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    for t in range(num_generate_tokens):
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token_function(
                model,
                input_ids=input_ids,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            pos_ += 1
            position_ids = pos_
            new_tokens.append(next_token.clone())
            input_ids = next_token.clone()

            if (pos_ - pixnum + 1) % (actnum + pixnum) == 0 and t+2 < num_generate_tokens:
                action = kwargs["action"][ (t+2) // pixnum ]
                input_ids = torch.cat((input_ids, action), dim=-1)
                position_ids = torch.tensor([pos_ + _ for _ in range(actnum+1)], dtype=torch.long, device="cuda")
                pos_ += actnum

    return new_tokens

def decode_n_tokens_for_gradio(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_one_token_function=decode_one_token,
):
    """
    Decode n tokens from the autoregressive model.
    Adapted from https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
    """
    new_tokens = []
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    position_id = position_ids[-1].unsqueeze(0)
    assert num_generate_tokens % 336 == 1, "should be pixnum x n + 1 to fill kvcache"
    for t in range(num_generate_tokens):
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token_function(
                model,
                input_ids=input_ids,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            position_id += 1
            position_ids = position_id
            new_tokens.append(next_token.clone())
            input_ids = next_token.clone()
    return new_tokens[:-1], position_id

def prefill(
    model,
    input_ids: torch.Tensor = None,
    position_ids: torch.Tensor = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.8,
    **kwargs,
):
    logits = model(input_ids=input_ids, position_ids=position_ids)
    # Only top-p or top-k can be provided
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"
    if top_p is not None:
        return sample_top_p(logits, temperature=temperature, top_p=top_p)
    else:
        return sample_top_k(logits, temperature=temperature, top_k=top_k)

def img_diagd_prepare_inputs(
    ongoing_row_list,
    row_token_num,
    ongoing_input,
    prompt,
    imagenum,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    promptlen: int = 347,
    **kwargs
):
    position_ids = []
    
    for i in ongoing_row_list:
        global_idx = promptlen + i * columnnum + row_token_num[i] - 1 + (imagenum - 1) * (pixnum + actnum)
        position_ids.append(global_idx)

    if row_token_num[ongoing_row_list[-1]] == 0:
        append_policy = kwargs.get("append_policy", True)
        if append_policy:
            idx_in_input_ids = ongoing_row_list[-1] * columnnum - 1
            ongoing_input.append(prompt[:, idx_in_input_ids].unsqueeze(-1))
        else:
            ongoing_input.append(ongoing_input[-1])

    input_ids = torch.cat(ongoing_input, dim=1)
    position_ids = torch.tensor(position_ids, device="cuda")

    return input_ids, position_ids

def img_diagd_decode_n_tokens(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
    promptlen: int = 347,
    **kwargs,
):
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    imagenum = 1
    cur_len = 1
    num_generate_tokens += 1
    prompt = kwargs.pop("prompt", None) 
    new_tokens = [input_ids.clone()]
    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
    row_token_num[0] += 1 
    ongoing_row_list = [0]
    ongoing_input = [input_ids.clone()]

    while True:
        if cur_len >= num_generate_tokens:
            break

        if cur_len % pixnum == 0 :#and image_start_token_id_index is None: 
            imagenum += 1
            action = kwargs["action"][cur_len // pixnum]
            ongoing_input.append(action)
            input_id = torch.cat(ongoing_input, dim=-1)
            position_ids = torch.arange(imagenum * (pixnum+actnum) - actnum - 1, imagenum * (pixnum+actnum), device="cuda")

        image_token_num = cur_len % pixnum

        if image_token_num == 1 and row_token_num[0] == windowsize:
            ongoing_row_list.append(1)

        if image_token_num >= 1:
            input_id, position_ids = img_diagd_prepare_inputs(ongoing_row_list=ongoing_row_list, ongoing_input = ongoing_input, imagenum=imagenum, row_token_num=row_token_num, promptlen=promptlen, prompt=prompt,**kwargs)  
            
        num_new_tokens = input_id.shape[1] if len(ongoing_row_list) > 0 else 1
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_some_token_function(
                model,
                input_ids=input_id,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            ongoing_input = []
        if len(ongoing_row_list) == 0:
            cur_len += 1
            ongoing_input.append(next_token[:,-1].clone())
            new_tokens.append(next_token[:,-1].clone())
            ongoing_row_list.append(0)
            row_token_num[0] += 1 
        else:
            need_remove_row = None
            cur_len += num_new_tokens
            for i in range(num_new_tokens):
                position_in_new_tokens = torch.sum(row_token_num[:(ongoing_row_list[i] + 1)], dim=0) + (imagenum - 1) * pixnum 
                new_tokens.insert(position_in_new_tokens, next_token[:,i].clone())
                ongoing_input.append(next_token[:,i].clone())
                row_token_num[ongoing_row_list[i]] += 1

                if row_token_num[ongoing_row_list[i]] == windowsize and ongoing_row_list[i] < rownum - 1:
                    ongoing_row_list.append(ongoing_row_list[i]+1)

                elif ongoing_row_list[i] == rownum - 1 and row_token_num[ongoing_row_list[i]] == columnnum:
                    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
                    ongoing_row_list = []
                    ongoing_input = [next_token[:,i]]
                    need_remove_row = None
                    break

                if row_token_num[ongoing_row_list[i]] == columnnum: ## this row is done
                    ongoing_input.pop()
                    need_remove_row = ongoing_row_list[i]

            if need_remove_row is not None:
                ongoing_row_list.remove(need_remove_row)
    return new_tokens

def img_diagd_prepare_inputs_for_gradio(
    ongoing_row_list,
    row_token_num,
    ongoing_input,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    promptlen: int = 347,
):
    position_ids = []
    
    for i in ongoing_row_list:
        global_idx = promptlen + i * columnnum + row_token_num[i] - 1
        position_ids.append(global_idx)

    if row_token_num[ongoing_row_list[-1]] == 0:
        ongoing_input.append(ongoing_input[-1])

    input_ids = torch.cat(ongoing_input, dim=1)
    position_ids = torch.tensor(position_ids, device="cuda")

    return input_ids, position_ids

def img_diagd_decode_n_token_for_gradio(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
):
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    cur_len = 0
    promptlen = position_ids[-1] + 1
    
    new_tokens = []
    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
    ongoing_row_list = []
    ongoing_input = []
    
    while True:
        if cur_len == num_generate_tokens:
            break

        image_token_num = cur_len

        if image_token_num == 1 and row_token_num[0] == windowsize:
            ongoing_row_list.append(1)
        if image_token_num == 0:
            input_id = input_ids

        if image_token_num >=1:
            input_id, position_ids = img_diagd_prepare_inputs_for_gradio(ongoing_row_list=ongoing_row_list, ongoing_input = ongoing_input, row_token_num=row_token_num, promptlen=promptlen)  
            
        num_new_tokens = input_id.shape[1] if len(ongoing_row_list) > 0 else 1
        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_some_token_function(
                model,
                input_ids=input_id,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            ongoing_input = []
        if len(ongoing_row_list) == 0:
            cur_len += 1
            ongoing_input.append(next_token[:,-1].clone())
            new_tokens.append(next_token[:,-1].clone())
            ongoing_row_list.append(0)
            row_token_num[0] += 1 
        else:
            need_remove_row = None
            cur_len += num_new_tokens
            for i in range(num_new_tokens):
                position_in_new_tokens = torch.sum(row_token_num[:(ongoing_row_list[i] + 1)], dim=0)
                new_tokens.insert(position_in_new_tokens, next_token[:,i].clone())
                ongoing_input.append(next_token[:,i].clone())
                row_token_num[ongoing_row_list[i]] += 1

                if row_token_num[ongoing_row_list[i]] == windowsize and ongoing_row_list[i] < rownum - 1:
                    ongoing_row_list.append(ongoing_row_list[i]+1)

                elif ongoing_row_list[i] == rownum - 1 and row_token_num[ongoing_row_list[i]] == columnnum:
                    row_token_num = torch.zeros((rownum,), dtype=torch.long, device="cuda")
                    ongoing_row_list = []
                    ongoing_input = [next_token[:,i]]
                    need_remove_row = None
                    break

                if row_token_num[ongoing_row_list[i]] == columnnum: ## this row is done
                    ongoing_input.pop()
                    need_remove_row = ongoing_row_list[i]

            if need_remove_row is not None:
                ongoing_row_list.remove(need_remove_row)

    with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            _ = decode_some_token_function(
                model,
                input_ids=next_token[:,-1],
                position_ids=position_ids+1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
    
    return new_tokens, position_ids+2



def vid_diagd_prepare_inputs(
    ongoing_row_list_v,
    row_token_num_v,
    ongoing_input_v,
    prompt,
    pixnum: int = 336,
    actnum: int = 11,
    rownum: int = 14,
    columnnum: int = 24,
    promptlen: int = 347,
    **kwargs
):
    new_frame = False
    position_ids = []

    for i in ongoing_row_list_v:
        global_idx = promptlen + i * columnnum + row_token_num_v[i // rownum][i % rownum] -1 + (i // rownum) * actnum
        position_ids.append(global_idx)

    lastrow = ongoing_row_list_v[-1]
    if lastrow % rownum == 0 and row_token_num_v[lastrow // rownum][lastrow % rownum] == 0:
        # WARNING
        action = kwargs["action"][lastrow // rownum]
        ongoing_input_v.append(action)
        position_ids.pop()
        pos_act = torch.arange( promptlen + (lastrow // rownum) * (pixnum+actnum) - actnum, promptlen + (lastrow // rownum) * (pixnum+actnum), device="cuda")
        position_ids.extend(pos_act.unbind())
        new_frame = True
    elif row_token_num_v[lastrow // rownum][lastrow % rownum] == 0:
        append_policy = kwargs.get("append_policy", True)
        if append_policy:
            idx_in_input_ids = (lastrow % rownum) * columnnum - 1
            ongoing_input_v.append(prompt[:, idx_in_input_ids].unsqueeze(-1))
        else:
            ongoing_input_v.append(ongoing_input_v[-1])

    input_ids = torch.cat(ongoing_input_v, dim=1)
    position_ids = torch.tensor(position_ids, device="cuda")

    return input_ids, position_ids, new_frame

def video_diagd_decode_n_tokens(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    num_generate_tokens: int,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.8,
    top_k: Optional[int] = None,
    decode_some_token_function=decode_some_token,
    pixnum: int = 336,
    actnum: int = 11,
    columnnum: int = 24,
    rownum: int = 14,
    windowsize: int = 2,
    promptlen: int = 347,
    **kwargs,
):
    assert (
        top_p is None or top_k is None
    ), "Only one of top-p or top-k can be provided, got top-p={top_p} and top-k={top_k}"

    cur_len = 1
    num_generate_tokens += 1
    prompt = kwargs.pop("prompt", None) 
    new_tokens = [input_ids.clone()]
    row_token_num_v = []
    ongoing_row_list_v = [0]
    row_token_num_v.append(torch.zeros((rownum,), dtype=torch.long, device="cuda"))
    row_token_num_v[0][0] += 1
    if row_token_num_v[0][0] == windowsize:
        ongoing_row_list_v.append(1)

    ongoing_input_v = [input_ids.clone()]

    while True:
        if cur_len >= num_generate_tokens:
            break


        input_id, position_ids, new_frame = vid_diagd_prepare_inputs(ongoing_row_list_v=ongoing_row_list_v, ongoing_input_v = ongoing_input_v, row_token_num_v=row_token_num_v, promptlen=promptlen, prompt=prompt, **kwargs)  
            
        num_new_tokens = input_id.shape[1]

        with torch.nn.attention.sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_some_token_function(
                model,
                input_ids=input_id,
                position_ids=position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            ongoing_input_v = []
            if new_frame:
                next_token = torch.cat([next_token[:,:-actnum], next_token[:,-1:]], dim=1)
                num_new_tokens = num_new_tokens - actnum + 1
            
        need_remove_row = None

        cur_len += num_new_tokens
        for i in range(num_new_tokens):
            last_frame = torch.stack(row_token_num_v[:ongoing_row_list_v[i] // rownum]).sum() if ongoing_row_list_v[i] // rownum > 0 else torch.tensor(0, dtype=torch.long, device="cuda")
            position_in_new_tokens = last_frame + torch.sum(row_token_num_v[ongoing_row_list_v[i] // rownum][:(ongoing_row_list_v[i] % rownum + 1)], dim=0)
                    
            new_tokens.insert(position_in_new_tokens, next_token[:,i].clone())
            ongoing_input_v.append(next_token[:,i].clone())
            row_token_num_v[ongoing_row_list_v[i] // rownum][ongoing_row_list_v[i] % rownum] += 1

            # WARNING
            if row_token_num_v[ongoing_row_list_v[i] // rownum][ongoing_row_list_v[i] % rownum] == windowsize and ongoing_row_list_v[i] < rownum * (num_generate_tokens//pixnum) - 1:
                ongoing_row_list_v.append(ongoing_row_list_v[i]+1)
                if ongoing_row_list_v[-1] % rownum == 0:
                    row_token_num_v.append(torch.zeros((rownum,), dtype=torch.long, device="cuda"))
            if row_token_num_v[ongoing_row_list_v[i] // rownum][ongoing_row_list_v[i] % rownum] == columnnum:
                ongoing_input_v.pop()
                need_remove_row = ongoing_row_list_v[i]

        if need_remove_row is not None:
            ongoing_row_list_v.remove(need_remove_row)
    return new_tokens

