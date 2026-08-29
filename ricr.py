"""PANINI DAG RICR scaffold. Complete the marked functions in Question 8."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence
import itertools

PLACEHOLDER_PATTERN = re.compile(r"<ENTITY_Q(\d+)>")


@dataclass(frozen=True)
class Candidate:
    qa_uid: str
    answer_names: tuple[str, ...]
    score: float
    question: str = ""
    answer_ids: tuple[str, ...] = ()
    answer_role_states: tuple[str, ...] = ()
    document_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainState:
    steps: tuple[Candidate, ...]
    answers_by_step: Mapping[int, str | tuple[str, ...]]
    score: float
    last_hop_score: float = 0.0

    @property
    def current_answers(self) -> tuple[str, ...]:
        return self.steps[-1].answer_names if self.steps else ()


@dataclass(frozen=True)
class RICRResult:
    components: tuple[tuple[int, ...], ...]
    chains: tuple[ChainState, ...]
    evidence: tuple[Candidate, ...]
    issued_queries: tuple[str, ...]
    fallback: bool = False


def normalize_entity_name(value: str) -> str:
    return " ".join(value.casefold().split())


def geometric_mean(scores: Sequence[float], epsilon: float = 1e-12) -> float:
    if not scores:
        return 0.0
    safe = [max(float(score), epsilon) for score in scores]
    return float(math.exp(sum(math.log(score) for score in safe) / len(safe)))


def panini_chain_score(steps: Sequence[Candidate]) -> float:
    normalized = [max(1e-6, min(1.0, 0.5 * (step.score + 1.0))) for step in steps]
    return geometric_mean(normalized, epsilon=1e-6)


def harmonic_mean(scores: Sequence[float]) -> float:
    valid = [float(score) for score in scores if float(score) > 1e-6]
    return len(valid) / sum(1.0 / score for score in valid) if valid else 1e-6


# def instantiate_question(
#     template: str,
#     answers_by_step: Mapping[int, str | tuple[str, ...]],
# ) -> str:
#     def replace(match: re.Match[str]) -> str:
#         step = int(match.group(1))
#         if step not in answers_by_step:
#             raise KeyError(f"Question references unresolved Q{step}: {template}")
#         value = answers_by_step[step]
#         return ", ".join(value) if isinstance(value, tuple) else value

#     return PLACEHOLDER_PATTERN.sub(replace, template)
def instantiate_question(
    template: str,
    answers_by_step: Mapping[int, str | tuple[str, ...]],
) -> str:
    def replace(match: re.Match[str]) -> str:
        step = int(match.group(1))
        if step not in answers_by_step:
            raise KeyError(f"Question references unresolved Q{step}: {template}")
        value = answers_by_step[step]
        # Handle tuples, lists, or single values safely and return a string
        if isinstance(value, (tuple, list)):
            return ", ".join(str(v) for v in value)
        return str(value)

    return PLACEHOLDER_PATTERN.sub(replace, template)

def identify_retrieval_components(
    decomposed_questions: Sequence[Mapping[str, object]],
) -> list[list[int]]:
    """Return connected retrieval components in deterministic topological order.

    TODO(student): build dependency and reverse-dependency maps from every
    ``<ENTITY_Qn>`` reference, find weakly connected components, detect cycles,
    and omit singleton components because they use PANINI's fallback.
    """

    n = len(decomposed_questions)
    if n <= 1:
        return []

    # Build adjacency list
    graph = {i: set() for i in range(1, n + 1)}
    in_degree = {i: 0 for i in range(1, n + 1)}

    for i, row in enumerate(decomposed_questions, start=1):
        q_text = str(row.get("question", ""))
        #print(q_text)
        refs = [int(m) for m in PLACEHOLDER_PATTERN.findall(q_text)]
        for ref in refs:
            if ref != i:
                graph[ref].add(i)
                in_degree[i] += 1

    # Find connected components using undirected edges
    adj_undirected = {i: set() for i in range(1, n + 1)}
    for u, v_set in graph.items():
        for v in v_set:
            adj_undirected[u].add(v)
            adj_undirected[v].add(u)

    visited = set()
    components = []

    for i in range(1, n + 1):
        if i not in visited and row_requires_retrieval(decomposed_questions[i - 1]):
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj_undirected[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(comp) > 1:
                # Sort component topologically based on node indices
                comp.sort()
                components.append(comp)
    #print(components)
    return components

    #raise NotImplementedError("Implement retrieval DAG identification")

def row_requires_retrieval(row: Mapping[str, object]) -> bool:
    return bool(row.get("requires_retrieval", True))

def run_panini_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    original_question: str,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
    multi_dependency_threshold: float = 0.3,
    unique_intermediate_entities: bool = True,
) -> RICRResult:
    """Execute PANINI over retrieval DAGs.

    TODO(student): implement the complete Question 8 contract:

    * topologically process each component;
    * substitute every parent answer before retrieval;
    * for multiple parents, score the Cartesian products by harmonic mean,
      retain at most ``beam_width`` products above the threshold, and retain
      the best product as a fallback when none passes;
    * at intermediate hops, retain the best state per stable answer ID (or
      normalized name when no ID exists), then prune to ``beam_width``;
    * at the final hop, expand and prune QA pairs directly without entity
      grouping; and
    * deduplicate evidence from every surviving final beam.

    A plan with no multi-question component retrieves ``original_question``
    directly. Do not treat document-local IDs such as ``e1`` as global IDs.
    """

    components = identify_retrieval_components(decomposed_questions)
    
    # Fallback if no multi-question component exists
    if not components:
        issued_q = original_question
        cands = list(retrieve_and_score(issued_q, candidates_per_hop))
        chains = tuple(
            ChainState(
                steps=(c,),
                answers_by_step={1: c.answer_names},
                score=panini_chain_score((c,)),
                last_hop_score=c.score,
            )
            for c in cands[:beam_width]
        )
        return RICRResult(
            components=(),
            chains=chains,
            evidence=tuple(cands),
            issued_queries=(issued_q,),
            fallback=True,
        )

    all_chains = []
    all_evidence = []
    issued_queries_list = []

    # Determine children map to identify final hop nodes in components
    children_map = {i: set() for i in range(1, len(decomposed_questions) + 1)}
    for i, row in enumerate(decomposed_questions, start=1):
        for ref in [int(m) for m in PLACEHOLDER_PATTERN.findall(str(row.get("question", "")))]:
            children_map[ref].add(i)

    for comp in components:
        node_beams = {}  # maps node_id (1-indexed) -> list[ChainState]
        
        for node_idx in comp:
            row = decomposed_questions[node_idx - 1]
            template = str(row.get("question", ""))
            
            # Find parents referenced in template
            parents = [int(m) for m in PLACEHOLDER_PATTERN.findall(template)]
            is_final = len(children_map[node_idx]) == 0

            if not parents:
                query = template
                issued_queries_list.append(query)
                cands = list(retrieve_and_score(query, candidates_per_hop))
                all_evidence.extend(cands)
                
                beams = []
                for c in cands:
                    state = ChainState(
                        steps=(c,),
                        answers_by_step={node_idx: c.answer_names},
                        score=panini_chain_score((c,)),
                        last_hop_score=c.score,
                    )
                    beams.append(state)
                node_beams[node_idx] = beams[:beam_width]

            elif len(parents) == 1:
                parent_id = parents[0]
                parent_beams = node_beams.get(parent_id, [])
                merged_beams = []
                
                for pb in parent_beams:
                    try:
                        query = instantiate_question(template, pb.answers_by_step)
                    except KeyError:
                        continue
                    issued_queries_list.append(query)
                    cands = list(retrieve_and_score(query, candidates_per_hop))
                    all_evidence.extend(cands)

                    for c in cands:
                        new_steps = pb.steps + (c,)
                        new_answers = dict(pb.answers_by_step)
                        new_answers[node_idx] = c.answer_names
                        score = panini_chain_score(new_steps)
                        merged_beams.append(
                            ChainState(
                                steps=new_steps,
                                answers_by_step=new_answers,
                                score=score,
                                last_hop_score=c.score,
                            )
                        )

                node_beams[node_idx] = prune_and_select_beams(
                    merged_beams, node_idx, is_final, beam_width, unique_intermediate_entities
                )

            else:
                # Multiple parents: Cartesian product and harmonic mean scoring
                parent_beam_lists = [node_beams.get(p, []) for p in parents]
                if any(not lst for lst in parent_beam_lists):
                    continue

                cartesian_tuples = []
                for combo in itertools.product(*parent_beam_lists):
                    scores = [state.score for state in combo]
                    h_score = harmonic_mean(scores)
                    cartesian_tuples.append((h_score, combo))

                cartesian_tuples.sort(key=lambda x: x[0], reverse=True)
                passing = [t for t in cartesian_tuples if t[0] >= multi_dependency_threshold]
                if not passing and cartesian_tuples:
                    passing = [cartesian_tuples[0]]
                selected_tuples = passing[:beam_width]

                merged_beams = []
                for h_score, combo in selected_tuples:
                    combined_answers = {}
                    combined_steps = ()
                    for pb in combo:
                        combined_answers.update(pb.answers_by_step)
                        combined_steps += pb.steps

                    try:
                        query = instantiate_question(template, combined_answers)
                    except KeyError:
                        continue
                    issued_queries_list.append(query)
                    cands = list(retrieve_and_score(query, candidates_per_hop))
                    all_evidence.extend(cands)

                    for c in cands:
                        new_steps = combined_steps + (c,)
                        new_answers = dict(combined_answers)
                        new_answers[node_idx] = c.answer_names
                        score = panini_chain_score(new_steps)
                        merged_beams.append(
                            ChainState(
                                steps=new_steps,
                                answers_by_step=new_answers,
                                score=score,
                                last_hop_score=c.score,
                            )
                        )

                node_beams[node_idx] = prune_and_select_beams(
                    merged_beams, node_idx, is_final, beam_width, unique_intermediate_entities
                )

        # Collect final component beams
        for node_idx in comp:
            if len(children_map[node_idx]) == 0 and node_idx in node_beams:
                all_chains.extend(node_beams[node_idx])

    # Deduplicate evidence from surviving final beams by qa_uid
    seen_uids = set()
    deduped_evidence = []
    for chain in all_chains:
        for step in chain.steps:
            if step.qa_uid not in seen_uids:
                seen_uids.add(step.qa_uid)
                deduped_evidence.append(step)

    return RICRResult(
        components=tuple(tuple(c) for c in components),
        chains=tuple(all_chains),
        evidence=tuple(deduped_evidence),
        issued_queries=tuple(issued_queries_list),
        fallback=False,
    )
    #raise NotImplementedError("Implement PANINI DAG RICR")

def prune_and_select_beams(
    beams: list[ChainState],
    node_idx: int,
    is_final: bool,
    beam_width: int,
    unique_intermediate_entities: bool,
) -> list[ChainState]:
    if not is_final and unique_intermediate_entities:
        entity_best = {}
        for b in beams:
            last_step = b.steps[-1]
            # Group by answer ID if present, otherwise normalized name
            key = last_step.answer_ids[0] if last_step.answer_ids else (last_step.answer_names[0] if last_step.answer_names else "unknown")
            if key not in entity_best or b.score > entity_best[key].score:
                entity_best[key] = b
        sorted_beams = sorted(
            entity_best.values(),
            key=lambda x: (x.score, x.last_hop_score),
            reverse=True,
        )
    else:
        sorted_beams = sorted(
            beams,
            key=lambda x: (x.score, x.last_hop_score),
            reverse=True,
        )
    return sorted_beams[:beam_width]

def run_linear_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
) -> list[ChainState]:
    first = next(
        (str(row["question"]) for row in decomposed_questions if row.get("requires_retrieval", True)),
        "",
    )
    return list(
        run_panini_ricr(
            decomposed_questions,
            retrieve_and_score,
            original_question=first,
            beam_width=beam_width,
            candidates_per_hop=candidates_per_hop,
        ).chains
    )
