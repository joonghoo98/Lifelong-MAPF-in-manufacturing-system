from agvtb.validate.events import event_graph, first_cycle


def k(sid, vid, t):
    return (sid, vid, round(t, 6))


def test_event_graph_catches_a_swap_that_the_node_graph_misses():  # DAIM tests/test_plan_validator.py:97-113
    paths = {1: [("S,X,0", 0.0), ("M,X,Y", 5.0), ("S,Y,0", 7.0)], 2: [("S,Y,0", 0.0), ("M,Y,X", 6.0), ("S,X,0", 8.0)]}
    nodes = {k(s, v, t) for v, p in paths.items() for s, t in p}
    one_way = [(k("S,X,0", 1, 0.0), k("S,X,0", 2, 8.0)), (k("S,X,0", 1, 0.0), k("M,Y,X", 2, 6.0))]
    other_way = [(k("S,Y,0", 2, 0.0), k("S,Y,0", 1, 7.0)), (k("S,Y,0", 2, 0.0), k("M,X,Y", 1, 5.0))]
    node_graph = {n: set() for n in nodes}
    for v, p in paths.items():
        for (s0, t0), (s1, t1) in zip(p, p[1:]):
            node_graph[k(s0, v, t0)].add(k(s1, v, t1))
    for a, b in one_way + other_way:
        node_graph[a].add(b)
    assert first_cycle(node_graph) is None
    graph, forever = event_graph(paths, {1: 0, 2: 0}, nodes, one_way + other_way)
    assert first_cycle(graph) is not None and forever == 0
    assert first_cycle(event_graph(paths, {1: 0, 2: 0}, nodes, one_way)[0]) is None  # one order only: 1 goes first


def test_releases_and_requests_follow_the_claim_rule():  # plan_validator.py:186-192
    p1 = [("S,a,0", 0.0), ("M,a,b", 0.0), ("S,b,0", 2.0), ("L,b,0", 2.0), ("S,b,0", 7.0), ("M,b,c", 7.0), ("S,c,0", 9.0)]
    p2 = [("S,x,0", 0.0), ("M,x,y", 1.0), ("S,y,0", 3.0), ("M,y,z", 3.0), ("S,z,0", 5.0)]
    paths = {1: p1, 2: p2}
    nodes = {k(s, v, t) for v, p in paths.items() for s, t in p}
    edges = [(k("L,b,0", 1, 2.0), k("S,y,0", 2, 3.0)),   # an L releases at its own end; an S after a move asks for the move
             (k("S,b,0", 1, 2.0), k("M,y,z", 2, 3.0)),   # an S releases at the end of the next move
             (k("M,x,y", 2, 1.0), k("S,b,0", 1, 7.0)),   # an S after an L asks for nothing
             (k("S,c,0", 1, 9.0), k("S,z,0", 2, 5.0))]   # from the last stop: a permanent block
    graph, forever = event_graph(paths, {1: 0, 2: 0}, nodes, edges)
    cross = {(a, b) for a, bs in graph.items() for b in bs if a[1] != b[1]}
    assert forever == 1 and cross == {(("e", 1, 3), ("s", 2, 1)), (("e", 1, 5), ("s", 2, 3))}
    assert graph[("s", 1, 1)] == {("e", 1, 1)} and graph[("e", 1, 1)] == {("s", 1, 3)}
    graph, _ = event_graph(paths, {1: 0, 2: 3}, nodes, edges)  # vehicle 2 has claimed up to index 3
    assert {(a, b) for a, bs in graph.items() for b in bs if a[1] != b[1]} == {(("e", 1, 5), ("s", 2, 3))}


def test_only_nodes_still_in_the_tapg_count_and_a_pruned_move_still_comes_before():  # final review M-3
    p1 = [("S,a,0", 0.0), ("M,a,b", 0.0), ("S,b,0", 2.0), ("M,b,c", 2.0), ("S,c,0", 4.0), ("L,c,0", 4.0),
          ("S,c,0", 4.0)]  # the last stop has the key of the stop before the load
    p2 = [("S,x,0", 0.0), ("M,x,y", 1.0), ("S,y,0", 3.0), ("M,y,z", 3.0), ("S,z,0", 5.0)]
    paths = {1: p1, 2: p2}
    nodes = {k(s, v, t) for v, p in paths.items() for s, t in p} - {k("S,a,0", 1, 0.0), k("M,a,b", 1, 0.0)}  # pruned
    edges = [(k("M,x,y", 2, 1.0), k("S,b,0", 1, 2.0)),   # the stop asks for the move before it, gone from the TAPG
             (k("M,y,z", 2, 3.0), k("S,c,0", 1, 4.0))]   # the later index of a twice-seen key: a stop after an L
    graph, forever = event_graph(paths, {1: 0, 2: 0}, nodes, edges)
    assert {(a, b) for a, bs in graph.items() for b in bs if a[1] != b[1]} == {(("e", 2, 1), ("s", 1, 1))}
    assert forever == 0 and graph[("s", 1, 3)] == {("e", 1, 3)} and ("e", 1, 1) not in graph  # pruned: not chained


def test_first_cycle_returns_a_real_cycle_and_is_linear():  # DAIM B final review I-2
    assert first_cycle({1: {2}, 2: {3}, 3: set()}) is None
    g = {0: {1}, 1: {2}, 2: {3}, 3: {1, 4}, 4: set()}
    cyc = first_cycle(g)
    assert sorted(cyc) == [1, 2, 3] and all(cyc[(i + 1) % len(cyc)] in g[cyc[i]] for i in range(len(cyc)))
    assert first_cycle({i: {i + 1} for i in range(200_000)} | {200_000: set()}) is None
