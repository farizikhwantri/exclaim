# EXCLAIM
Explainable models for Requirements and non-compliance detection

Formulation

**Formal Problem Definition (Entailment Graph View).**

An **assurance case** can be modeled as a directed acyclic graph (DAG)

$$
G = (V, E)
$$

where:

* $V = C \cup A \cup E$ is the set of nodes, consisting of *claims* ($C$), *arguments* ($A$), and *evidence* ($E$);
* $E \subseteq V \times V$ is the set of directed edges, each representing a support relation (“justifies” or “supports”).

Each edge $(u, v) \in E$ is interpreted as a **local entailment** relation:

$$
u \models v \quad \text{(if true, node \(u\) provides sufficient justification for node \(v\))}.
$$

The **multi-hop inference problem for compliance detection** is then defined as:

* Given a regulatory requirement $r \in R$, identify a path

  $$
  P(r) = \langle v_1, v_2, \ldots, v_k \rangle
  $$

  in $G$ such that:

  1. $v_1$ is a claim directly addressing $r$;
  2. $ v_k \in E$ is an evidence node grounded in a textual artifact $ t \in T$;
  3. For all consecutive pairs $(v_j, v_{j+1})$, the entailment relation holds:

     $$
     v_j \models v_{j+1}.
     $$

* Compliance holds if there exists at least one path $P(r)$ such that all edges are entailed:

  $$
  \text{Compliant}(r) \iff \exists P(r) \;\; \text{s.t.} \;\; \forall (v_j, v_{j+1}) \in P(r), \; v_j \models v_{j+1}.
  $$

* Non-compliance occurs if no such fully-entailed path exists:

  $$
  \text{Non-Compliant}(r) \iff \forall P(r), \; \exists (v_j, v_{j+1}) \in P(r) \;\; \text{s.t.} \;\; v_j \not\models v_{j+1}.
  $$
