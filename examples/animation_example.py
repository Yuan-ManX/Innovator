from innovator.animation_agent import (
    AnimationAgent,
    AnimationContext,
    GlobalStyle,
    CharacterProfile,
    PlanningStage,
    StoryboardStage,
    MotionDesignStage,
    RenderStage,
    VideoRenderer
)
import asyncio


async def main():
    llm = LLMClient() 
    renderer = VideoRenderer()

    context = AnimationContext(
        style=GlobalStyle(
            visual_style="anime cinematic",
            lighting="dramatic",
            color_palette="cold blue",
        ),
        characters={
            "hero": CharacterProfile(
                name="Hero",
                appearance="young, athletic",
                costume="black coat",
                personality="calm, determined"
            )
        }
    )

    agent = AnimationAgent(
        context=context,
        stages=[
            PlanningStage(llm),
            StoryboardStage(llm),
            MotionDesignStage(llm),
            RenderStage(renderer),
        ],
    )

    final_context = await agent.run()

    # 输出渲染结果
    for scene in final_context.timeline.scenes:
        for shot in scene.shots:
            print(f"Shot {shot.id} rendered at: {getattr(shot, 'render_output', 'not rendered')}")


asyncio.run(main())
