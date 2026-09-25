"""修订稿重新打时间轴核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

输入两侧词序列：原稿 words（按时间排列、相邻不重叠，至多 300 词）与修订词
revisions（只需 text/speaker，按顺序排列，至多 300 个）。先在两侧之间求

* 匹配数最多；
* 匹配数相同时，匹配索引对序列 ``((i1,j1),(i2,j2),...)`` 字典序最小

的保序匹配（i 为原词序号、j 为修订词序号，两侧各自严格递增），再据此为每个
修订词确定 start_ms/end_ms。

匹配（最长公共子序列，LCS）
---------------------------

以 ``(text, speaker)`` 为符号。后向 DP：``lcs[i][j]`` 为原稿后缀 ``[i,n)`` 与
修订词后缀 ``[j,m)`` 的 LCS 长度；两符号相等时取对角匹配（长度上严格不劣于
跳过任一侧），不等时取“跳过原词 / 跳过修订词”两侧的较大值。

字典序最小解不必在 DP 状态里保存整条序列：长度表建好后，从状态 ``(r,c)`` 贪心
挑选**最小的可行首对**——按 i 升序、j 升序找到第一个满足

    orig[i] == rev[j] 且 1 + lcs[i+1][j+1] == lcs[r][c]

的索引对。跳过 ``[r,i)`` 与 ``[c,j)`` 不产生任何匹配，总能做到；条件又保证
配上该对后后缀恰好补满最优长度，故它是某个最优匹配的首对，且枚举顺序保证
字典序最小。取定后状态推进到 ``(i+1,j+1)`` 重复。整个重建中 i、j 都只增不减，
总扫描量 O(n·m)。

时间分配
--------

已匹配修订词直接沿用原词的 start_ms/end_ms。未匹配修订词按连续段（run）切分：
首条匹配之前、相邻匹配之间、末条匹配之后各为一段。每段两侧的**锚点**为：

* 中间段：前一匹配原词的 end_ms 与后一匹配原词的 start_ms；
* 前缀段：原稿首词 start_ms（虚拟锚点）与首条匹配原词的 start_ms；
* 后缀段：末条匹配原词的 end_ms 与原稿末词 end_ms（虚拟锚点）；
* 完全无匹配：全部修订词作为一段，均分原稿首词 start_ms 到末词 end_ms 的全长。

设锚点间隔为 span、段内有 k 个词：span < k 即不足每词 1 毫秒，抛出
DomainError(code="infeasible_transcript_retime") 并指出修订词范围；否则每词
base = span // k 毫秒，余下 span % k 毫秒从前往后逐个加 1 毫秒。原稿时间有序
且不重叠，故任一锚点间隔均非负，分配结果天然有序、不重叠（允许端点相接）。
"""

from .errors import DomainError
from .segmentation import validate_words


def _fail(code, message, **extra):
    raise DomainError(code, message, **extra)


def _best_matches(orig, revised):
    """返回字典序最小的最大匹配，元素为 (原词序号, 修订词序号)。"""
    n = len(orig)
    m = len(revised)

    # ---- 后向 LCS 长度 DP：lcs[i][j] 为两侧后缀的 LCS 长度 ----
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row = lcs[i]
        next_row = lcs[i + 1]
        symbol = orig[i]
        for j in range(m - 1, -1, -1):
            if symbol == revised[j]:
                row[j] = next_row[j + 1] + 1
            elif next_row[j] >= row[j + 1]:
                row[j] = next_row[j]
            else:
                row[j] = row[j + 1]

    # ---- 贪心重建：每步取最小的可行首对（i 升序、j 升序）----
    pairs = []
    r, c = 0, 0
    remaining = lcs[0][0]
    while remaining > 0:
        suffix_target = remaining - 1
        chosen = False
        for i in range(r, n):
            symbol = orig[i]
            for j in range(c, m):
                if symbol == revised[j] and lcs[i + 1][j + 1] == suffix_target:
                    pairs.append((i, j))
                    r, c = i + 1, j + 1
                    chosen = True
                    break
            if chosen:
                break
        remaining = suffix_target
    return pairs


def retime_transcript(words, revisions):
    """为修订词逐词重新打时间轴，返回 (retimed, matched_word_count)。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验）；revisions 按修订顺序排列，每项需具备
    text / speaker。

    retimed 与 revisions 等长、同序，每项为 dict：
    index / start_ms / end_ms / text / speaker / source_word_index。
    source_word_index 为命中的原稿词序号（0-based），未匹配为 None。

    某段连续未匹配修订词的锚点间隔不足每词 1 毫秒时抛出
    DomainError(code="infeasible_transcript_retime")（HTTP 422），附加
    revision_start_word / revision_end_word（0-based、含端点）、
    word_count 与 available_ms。
    """
    validate_words(words)
    n = len(words)
    m = len(revisions)

    orig = [(w.text, w.speaker) for w in words]
    revised = [(w.text, w.speaker) for w in revisions]

    pairs = _best_matches(orig, revised)

    starts = [0] * m
    ends = [0] * m
    sources = [None] * m
    for word_index, revision_index in pairs:
        starts[revision_index] = words[word_index].start_ms
        ends[revision_index] = words[word_index].end_ms
        sources[revision_index] = word_index

    def fill_run(lo, hi, left_ms, right_ms):
        """把修订词区间 [lo, hi) 均分到锚点间隔 [left_ms, right_ms]。"""
        k = hi - lo
        if k == 0:
            return
        span = right_ms - left_ms
        if span < k:
            _fail(
                "infeasible_transcript_retime",
                f"修订词第 {lo}～{hi - 1} 个（共 {k} 词）的相邻锚点间隔仅 "
                f"{span} 毫秒，不足每词至少 1 毫秒",
                revision_start_word=lo,
                revision_end_word=hi - 1,
                word_count=k,
                available_ms=span,
            )
        base, extra = divmod(span, k)  # 余下 extra 毫秒从前往后逐个加 1
        cursor = left_ms
        for p in range(k):
            duration = base + (1 if p < extra else 0)
            starts[lo + p] = cursor
            ends[lo + p] = cursor + duration
            cursor += duration

    if not pairs:
        # 无任何匹配：全部修订词均分原稿全长
        fill_run(0, m, words[0].start_ms, words[n - 1].end_ms)
    else:
        first_i, first_j = pairs[0]
        fill_run(0, first_j, words[0].start_ms, words[first_i].start_ms)
        for (prev_i, prev_j), (next_i, next_j) in zip(pairs, pairs[1:]):
            fill_run(
                prev_j + 1,
                next_j,
                words[prev_i].end_ms,
                words[next_i].start_ms,
            )
        last_i, last_j = pairs[-1]
        fill_run(last_j + 1, m, words[last_i].end_ms, words[n - 1].end_ms)

    retimed = [
        {
            "index": j,
            "start_ms": starts[j],
            "end_ms": ends[j],
            "text": revisions[j].text,
            "speaker": revisions[j].speaker,
            "source_word_index": sources[j],
        }
        for j in range(m)
    ]
    return retimed, len(pairs)
