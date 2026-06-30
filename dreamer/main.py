# SOURCE: NaturalDreamer (https://github.com/InexperiencedMe/NaturalDreamer) — da adattare
import torch
import argparse
import os
import sys
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import CleanEnvWrapper
from utils      import saveLossesToCSV, ensureParentFolders

# Allow imports from project root (limo_env.py, reward.py) — same pattern as modal_train_ppo.py
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from limo_env import LimoCustomEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase)

    # TODO: add curriculum logic (stages 0→1→2) matching train_ppo.py's CurriculumCallback pattern
    # For now, start at curriculum stage 0 (roadmap §4.2): 2-4 obstacles, r∈[0.10, 0.15]
    _env_kwargs = dict(n_obstacles_range=(2, 4), obs_radius_range=(0.10, 0.15), randomize_goal=True)
    env           = CleanEnvWrapper(LimoCustomEnv(**_env_kwargs))
    envEvaluation = CleanEnvWrapper(LimoCustomEnv(**_env_kwargs))

    observationShape = env.observation_space.shape        # (23,)
    actionSize       = env.action_space.shape[0]           # 2
    actionLow        = env.action_space.low.tolist()       # [0.0, -1.0]
    actionHigh       = env.action_space.high.tolist()      # [1.0,  1.0]
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}")

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config.dreamer)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    for _ in range(iterationsNum):
        for _ in range(config.replayRatio):
            sampledData                         = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData)
            behaviorMetrics                     = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True)
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward": mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="limo-dreamer.yml")
    main(parser.parse_args().config)
