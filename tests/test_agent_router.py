"""
Test Script for AgentRouter
Demonstrates routing with confidence scoring and fallback
"""

from innovator.agent_router import (
    AgentRouter,
    AgentType,
    Task,
    animation_agent_score,
    film_agent_score,
    game_agent_score,
)

def main():
    # Initialize router with confidence threshold and fallback
    router = AgentRouter(confidence_threshold=0.65, fallback_agent=AgentType.PLANNER)

    # Register execution agents with scoring functions
    router.register_agent(AgentType.ANIMATION, animation_agent_score)
    router.register_agent(AgentType.FILM, film_agent_score)
    router.register_agent(AgentType.GAME, game_agent_score)

    # Example task
    task = Task(
        task_id="task_001",
        task_type="animation",
        payload={"prompt": "Create a cinematic sword fight animation"},
    )

    current_agent = None
    context = {}

    print("\n=== Starting Agent Routing Simulation ===\n")

    while True:
        decision = router.route(current_agent, task, context)
        print(f"{current_agent} → {decision}")

        if decision.next_agents[0] == AgentType.END:
            print("\n=== Pipeline Complete ===\n")
            break

        # Move to next agent
        current_agent = decision.next_agents[0]

        # Simulate render review if Render agent reached
        if current_agent == AgentType.RENDER:
            # For testing, we can toggle between 'accept', 'revise', 'redesign'
            context["review_result"] = "accept"  # change to test iterative loops

if __name__ == "__main__":
    main()
