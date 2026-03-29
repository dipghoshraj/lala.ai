## Query Processing & Planning Sequence

This diagram focuses on how the user prompt is prepared and planned before retrieval.

```mermaid
sequenceDiagram
    participant User
    participant Application
    participant SessionContextBuilder as Session Context Builder
    participant QueryRewriter
    participant ReasoningAgent as Reasoning Agent (Task Planner)

    User->>Application: User Prompt

    Application->>SessionContextBuilder: Build session context
    SessionContextBuilder-->>Application: Context

    Application->>QueryRewriter: Rewrite / optimize query
    QueryRewriter-->>Application: Optimized query

    Application->>ReasoningAgent: Plan task
    ReasoningAgent-->>Application: Retrieval plan
    ```