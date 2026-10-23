from agvtb.planning.safe_tail import tail_cut_index

P = [("S,a,0", 0.0), ("M,a,b", 0.0), ("S,b,0", 2.0), ("L,b,0", 2.0), ("S,b,0", 12.0), ("M,b,c", 12.0), ("S,c,0", 14.0),
     ("M,c,d", 14.0), ("S,d,0", 16.0)]  # the last L is index 3


def test_the_splice_is_the_first_stop_after_the_last_load_and_the_claimed_part():  # DAIM spec §3.2, §8.1-1
    assert tail_cut_index(P, 3, 4, 5.0) == 5       # (가) the unload in progress: the stop right after it
    assert tail_cut_index(P, 1, 2, 1.0) == 5       # the previous task's load stays, whatever comes before it
    assert tail_cut_index(P, 5, 6, 12.5) == 7      # (나) on the way to c: the stop at the end of the claimed move
    assert tail_cut_index(P, 6, 6, 14.0) == 7      # standing at c: c itself
    assert tail_cut_index(P, 7, 8, 14.5) == -1     # claimed up to the last state
    assert tail_cut_index(P[:5], 3, 4, 5.0) == -1  # no tail left after the stop that follows the load


def test_a_margin_pushes_the_splice_back():  # DAIM spec §3.2 (m), §5
    assert tail_cut_index(P, 3, 4, 5.0, margin=8.0) == 7    # leaving b at 12 < 5 + 8: not b; c is left at 14
    assert tail_cut_index(P, 3, 4, 5.0, margin=10.0) == -1
