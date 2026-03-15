import torch


class TriangularCausalMask():
    def __init__(self, batch_size, sequence_length, device="cpu"):
        ## 瀹氫箟鎺╃爜褰㈢姸
        mask_shape = [batch_size, 1, sequence_length, sequence_length]
        ## 绂佺敤姊害璁＄畻锛屼互鍑忓皯璁＄畻閲?
        with torch.no_grad():
            ## 鐢熸垚涓婁笁瑙掔煩闃碉紝骞跺皢鍏惰浆鎹负甯冨皵鍨嬪紶閲?
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    ## 杩斿洖鎺╃爜
    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, batch_size, num_heads, sequence_length, index, scores, device="cpu"):
        ## 鐢熸垚鍏ㄤ负 True 鐨勪笂涓夎鐭╅樀
        base_mask = torch.ones(sequence_length, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        ## 鎵╁睍鐭╅樀褰㈢姸
        mask_expanded = base_mask[None, None, :].expand(
            batch_size,
            num_heads,
            sequence_length,
            scores.shape[-1],
        )
        ## 鐢熸垚涓€涓寚绀虹煩闃碉紝琛ㄧず鍝簺浣嶇疆闇€瑕佷繚鐣欙紝鍝簺浣嶇疆闇€瑕佹帺鐩?
        indicator = mask_expanded[torch.arange(batch_size)[:, None, None],
                                  torch.arange(num_heads)[None, :, None],
                                  index, :].to(device)
        ## 閲嶆柊璋冩暣褰㈢姸
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


if __name__ == '__main__':
    # 瀹炰緥鍖?TriangularCausalMask 绫?
    batch_size = 2
    sequence_length = 4  # sen_length
    mask = TriangularCausalMask(batch_size, sequence_length)

    # 杈撳嚭鎺╃爜鐨勫舰鐘?
    print(mask.mask.shape)

    # 瀹炰緥鍖?ProbMask 绫?
    num_heads = 2  # H 鏄寚娉ㄦ剰鍔涘ご鐨勬暟閲忥紝
    index = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])  # batch_szie 2 sen_length/time_step 4
    # index鏄竴涓舰鐘朵负 (B, H, L) 鐨勫紶閲忥紝鍏朵腑 B 鏄?batch size锛孒 鏄敞鎰忓姏澶存暟锛孡 鏄簭鍒楅暱搴︼紝鐢ㄦ潵琛ㄧず褰撳墠娉ㄦ剰鍔涘ご鐨勬煡璇㈠悜閲忓湪搴忓垪涓殑浣嶇疆銆?
    scores = torch.randn(batch_size, num_heads, sequence_length, sequence_length)
    prob_mask = ProbMask(batch_size, num_heads, sequence_length, index, scores)

    # 杈撳嚭鎺╃爜鐨勫舰鐘?
    print(prob_mask.mask.shape)
