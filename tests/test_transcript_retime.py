"""修订稿重新打时间轴核心逻辑测试（纯标准库）。

包含：
* 随机小输入上与“枚举全部保序匹配”的暴力最优对拍——匹配数最大、
  同数取索引对序列字典序最小；
* 未匹配连续段的均分、余下毫秒前置分配、虚拟锚点、无匹配全长均分；
* 间隔不足每词 1 毫秒的 422（DomainError）与修订词范围；
* 词序/时间校验复用 validate_words。
"""

import random
import unittest
from itertools import combinations
from types import SimpleNamespace

from app.errors import DomainError
from app.transcript_retime import retime_transcript


def W(start, end, text, speaker="A"):
    return SimpleNamespace(start_ms=start, end_ms=end, text=text, speaker=speaker)


def R(text, speaker="A"):
    return SimpleNamespace(text=text, speaker=speaker)


def make_original(symbols, start=0, gap=100, dur=1000):
    """按符号列表生成有序、不重叠（端点可相接）的原稿词。

    匹配对拍只关心符号序列，时间轴给足余量（每词 1000ms、词间 100ms），
    保证任意未匹配连续段都有足够间隔，不至于触发 422。
    """
    words = []
    t = start
    for text, speaker in symbols:
        words.append(W(t, t + dur, text, speaker))
        t += dur + gap
    return words


def brute_force_best(orig, revised):
    """枚举全部保序匹配，返回字典序最小的最大匹配 (i,j) 元组；可为空元组。"""
    n, m = len(orig), len(revised)
    equal = [
        (i, j)
        for i in range(n)
        for j in range(m)
        if orig[i] == revised[j]
    ]
    best = ()
    # 全部等号对子集枚举：匹配即两侧索引各自严格递增的子集
    for size in range(len(equal), 0, -1):
        if size < len(best):
            break
        for chosen in combinations(equal, size):
            iseq = [p[0] for p in chosen]
            jseq = [p[1] for p in chosen]
            if iseq == sorted(iseq) and len(set(iseq)) == size and \
               jseq == sorted(jseq) and len(set(jseq)) == size:
                candidate = tuple(chosen)
                if size > len(best) or candidate < best:
                    best = candidate
        if best:
            break
    return best


class BestMatchDifferentialTests(unittest.TestCase):
    def test_matches_brute_force(self):
        random.seed(20260925)
        symbols = [("甲", "A"), ("乙", "A"), ("丙", "B"), ("丁", "B"),
                   ("甲", "B"), ("乙", "B"), ("戊", "A")]
        # 原稿首尾各加一对绝不会出现在 revisions 中的唯一哨兵词，使前缀/后缀
        # 未匹配段（若有）始终拿到正间隔，匹配对拍得以独立于 422 可行性。
        prefix = [("哨兵起", "Z")]
        suffix = [("哨兵止", "Z")]
        for trial in range(500):
            n_core = random.randint(1, 7)
            m = random.randint(1, 7)
            orig_core = [random.choice(symbols) for _ in range(n_core)]
            revised = [random.choice(symbols) for _ in range(m)]
            orig = prefix + orig_core + suffix
            words = make_original(orig, gap=1000)
            revisions = [R(t, s) for t, s in revised]
            retimed, matched_count = retime_transcript(words, revisions)
            expected = brute_force_best(orig_core, revised)
            # 哨兵把核心原词序号整体后移 1
            expected_shifted = tuple((i + 1, j) for i, j in expected)
            self.assertEqual(matched_count, len(expected_shifted), f"trial {trial}")
            got = tuple(
                (w["source_word_index"], w["index"])
                for w in retimed
                if w["source_word_index"] is not None
            )
            self.assertEqual(got, expected_shifted, f"trial {trial} tie-break")

    def test_all_identical_symbols_prefers_earliest_pairs(self):
        # 两侧同为 3 个 (甲,A)：最大匹配 3 对，字典序最小为 (0,0)(1,1)(2,2)
        words = make_original([("甲", "A")] * 3)
        revisions = [R("甲") for _ in range(3)]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 3)
        self.assertEqual([w["source_word_index"] for w in retimed], [0, 1, 2])

    def test_different_side_lengths_lex_min(self):
        # orig: 甲 甲 乙 ；revised: 甲 乙 甲
        # 长度 2 的匹配有 (0,0)(2,1) 与 (0,1)(2,2)，第一对相同，第二对
        # (2,1) < (2,2)，字典序最小为 (0,0)(2,1)
        orig = [("甲", "A"), ("甲", "A"), ("乙", "A")]
        words = make_original(orig)
        revisions = [R("甲"), R("乙"), R("甲")]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 2)
        # 配对为 (原0,修0) 与 (原1,修2)：按修订序记录 source_word_index
        sources = [(w["index"], w["source_word_index"]) for w in retimed]
        self.assertEqual(sources, [(0, 0), (1, None), (2, 1)])


class RetimingTests(unittest.TestCase):
    def test_matched_words_keep_original_times(self):
        words = make_original(
            [("大家好", "A"), ("今天", "A"), ("再见", "B")], dur=500, gap=100
        )
        revisions = [R("大家好", "A"), R("今天", "A"), R("再见", "B")]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 3)
        self.assertEqual(
            [(w["start_ms"], w["end_ms"], w["source_word_index"]) for w in retimed],
            [(0, 500, 0), (600, 1100, 1), (1200, 1700, 2)],
        )

    def test_unmatched_run_between_anchors_split_evenly_with_remainder_front(self):
        # 原稿：词0 [0,100] 甲，词1 [300,400] 乙；锚点间隔 100→300 共 200ms
        # 修订：甲 + x + y + z + 乙；中间 3 词均分 200ms => 67,67,66（余数前置）
        words = [W(0, 100, "甲"), W(300, 400, "乙")]
        revisions = [R("甲"), R("x"), R("y"), R("z"), R("乙")]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 2)
        self.assertEqual(
            [(w["start_ms"], w["end_ms"]) for w in retimed],
            [(0, 100), (100, 167), (167, 234), (234, 300), (300, 400)],
        )
        self.assertEqual(
            [w["source_word_index"] for w in retimed],
            [0, None, None, None, 1],
        )

    def test_prefix_and_suffix_use_virtual_anchors(self):
        # 原稿：零 [0,100)（未匹配）、甲 [200,300)、乙 [350,450)、尾 [450,550)（未匹配）
        # 修订：u v 甲 x 乙 p q
        # 匹配 (1,2) 甲、(2,4) 乙；
        # 前缀 2 词均分虚拟锚点 0→200：各 100ms；
        # 中间 1 词分 300→350：50ms；
        # 后缀 2 词均分 450→550（虚拟锚点=末词 end）：各 50ms。
        words = [
            W(0, 100, "零"),
            W(200, 300, "甲"),
            W(350, 450, "乙"),
            W(450, 550, "尾"),
        ]
        revisions = [R("u"), R("v"), R("甲"), R("x"), R("乙"), R("p"), R("q")]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 2)
        self.assertEqual(
            [(w["start_ms"], w["end_ms"]) for w in retimed],
            [
                (0, 100),
                (100, 200),
                (200, 300),
                (300, 350),
                (350, 450),
                (450, 500),
                (500, 550),
            ],
        )
        self.assertEqual(
            [w["source_word_index"] for w in retimed],
            [None, None, 1, None, 2, None, None],
        )

    def test_prefix_zero_span_infeasible(self):
        # 前缀虚拟锚点=原稿首词 start_ms=0，首匹配也是词0：span 0 却有未匹配词
        words = [W(0, 100, "甲"), W(100, 200, "乙")]
        revisions = [R("u"), R("甲"), R("乙")]
        with self.assertRaises(DomainError) as ctx:
            retime_transcript(words, revisions)
        self.assertEqual(ctx.exception.code, "infeasible_transcript_retime")
        self.assertEqual(ctx.exception.extra["revision_start_word"], 0)
        self.assertEqual(ctx.exception.extra["revision_end_word"], 0)
        self.assertEqual(ctx.exception.extra["word_count"], 1)
        self.assertEqual(ctx.exception.extra["available_ms"], 0)

    def test_suffix_zero_span_infeasible(self):
        words = [W(0, 100, "甲"), W(100, 200, "乙")]
        revisions = [R("甲"), R("乙"), R("z")]
        with self.assertRaises(DomainError) as ctx:
            retime_transcript(words, revisions)
        self.assertEqual(ctx.exception.code, "infeasible_transcript_retime")
        self.assertEqual(ctx.exception.extra["revision_start_word"], 2)
        self.assertEqual(ctx.exception.extra["revision_end_word"], 2)
        self.assertEqual(ctx.exception.extra["available_ms"], 0)

    def test_gap_smaller_than_word_count_infeasible_and_range(self):
        # 间隔 [100,102) 仅 2ms 要塞 3 个词
        words = [W(0, 100, "甲"), W(102, 202, "乙")]
        revisions = [R("甲"), R("a"), R("b"), R("c"), R("乙")]
        with self.assertRaises(DomainError) as ctx:
            retime_transcript(words, revisions)
        self.assertEqual(ctx.exception.code, "infeasible_transcript_retime")
        self.assertEqual(ctx.exception.extra["revision_start_word"], 1)
        self.assertEqual(ctx.exception.extra["revision_end_word"], 3)
        self.assertEqual(ctx.exception.extra["word_count"], 3)
        self.assertEqual(ctx.exception.extra["available_ms"], 2)

    def test_gap_exactly_word_count_gives_one_ms_each(self):
        words = [W(0, 100, "甲"), W(103, 203, "乙")]
        revisions = [R("甲"), R("a"), R("b"), R("c"), R("乙")]
        retimed, _ = retime_transcript(words, revisions)
        self.assertEqual(
            [(w["start_ms"], w["end_ms"]) for w in retimed[1:4]],
            [(100, 101), (101, 102), (102, 103)],
        )

    def test_no_matches_splits_full_span(self):
        # 原稿全长 [0,1000)；修订 3 词全不匹配：1000/3 => 334,333,333
        words = [W(0, 400, "甲"), W(400, 1000, "乙")]
        revisions = [R("x"), R("y"), R("z")]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 0)
        self.assertEqual(
            [(w["start_ms"], w["end_ms"], w["source_word_index"]) for w in retimed],
            [(0, 334, None), (334, 667, None), (667, 1000, None)],
        )

    def test_no_matches_full_span_too_small_infeasible(self):
        words = [W(0, 1, "甲"), W(1, 2, "乙")]  # 全长仅 2ms
        revisions = [R("x"), R("y"), R("z")]
        with self.assertRaises(DomainError) as ctx:
            retime_transcript(words, revisions)
        self.assertEqual(ctx.exception.code, "infeasible_transcript_retime")
        self.assertEqual(ctx.exception.extra["revision_start_word"], 0)
        self.assertEqual(ctx.exception.extra["revision_end_word"], 2)
        self.assertEqual(ctx.exception.extra["available_ms"], 2)
        self.assertEqual(ctx.exception.extra["word_count"], 3)

    def test_speaker_participates_in_match(self):
        # 原稿：零（A）[0,100)、甲（A）[200,300)、甲（B）[400,500)
        # 修订：甲（B）、甲（A）
        # 可行匹配各仅 1 对：(2,0) 与 (1,1)；字典序最小首对为 (1,1)
        words = [W(0, 100, "零", "A"), W(200, 300, "甲", "A"), W(400, 500, "甲", "B")]
        revisions = [R("甲", "B"), R("甲", "A")]
        retimed, count = retime_transcript(words, revisions)
        self.assertEqual(count, 1)
        self.assertEqual(
            [(w["index"], w["source_word_index"]) for w in retimed],
            [(0, None), (1, 1)],
        )

    def test_output_ordering_and_non_overlap(self):
        random.seed(42)
        vocab = [("甲", "A"), ("乙", "A"), ("丙", "B"), ("丁", "B")]
        feasible = 0
        for _ in range(200):
            n = random.randint(1, 10)
            t = 0
            words = []
            for _ in range(n):
                dur = random.randint(100, 500)
                text, speaker = random.choice(vocab)
                words.append(W(t, t + dur, text, speaker))
                t += dur + random.randint(0, 3)
            m = random.randint(1, 10)
            revisions = [R(*random.choice(vocab)) for _ in range(m)]
            try:
                retimed, count = retime_transcript(words, revisions)
            except DomainError as exc:
                # 锚点间隔不足是合法结果：范围必须指向一段连续未匹配修订词
                self.assertEqual(exc.code, "infeasible_transcript_retime")
                lo = exc.extra["revision_start_word"]
                hi = exc.extra["revision_end_word"]
                self.assertGreaterEqual(lo, 0)
                self.assertLessEqual(hi, m - 1)
                self.assertEqual(exc.extra["word_count"], hi - lo + 1)
                self.assertLess(exc.extra["available_ms"], hi - lo + 1)
                continue
            feasible += 1
            prev_end = 0
            for w in retimed:
                self.assertGreaterEqual(w["start_ms"], prev_end)
                self.assertGreater(w["end_ms"], w["start_ms"])
                self.assertLessEqual(w["end_ms"], words[-1].end_ms)
                prev_end = w["end_ms"]
            # 末条匹配即最后一个修订词时，时间轴止于该原词 end_ms（可早于原稿末词）；
            # 否则后缀段均分到原稿末词终点
            if retimed[-1]["source_word_index"] is None:
                self.assertEqual(prev_end, words[-1].end_ms)
            self.assertEqual(count, sum(w["source_word_index"] is not None for w in retimed))
            # 匹配词保留原 start_ms/end_ms
            for w in retimed:
                if w["source_word_index"] is not None:
                    src = words[w["source_word_index"]]
                    self.assertEqual(w["start_ms"], src.start_ms)
                    self.assertEqual(w["end_ms"], src.end_ms)
        self.assertGreater(feasible, 50)

    def test_word_validation_reused(self):
        words = [W(0, 100, "甲"), W(50, 150, "乙")]  # 重叠
        with self.assertRaises(DomainError) as ctx:
            retime_transcript(words, [R("甲")])
        self.assertEqual(ctx.exception.code, "overlapping_time_range")


if __name__ == "__main__":
    unittest.main()
