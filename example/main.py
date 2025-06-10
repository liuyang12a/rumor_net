import asyncio

import ray
from agentsociety.configs import (AgentConfig, AgentsConfig, Config, EnvConfig,
                                  LLMConfig, MapConfig)
from agentsociety.llm import LLMProviderType
from agentsociety.metrics import MlflowConfig
from agentsociety.simulation import AgentSociety
from agentsociety.storage import AvroConfig, PostgreSQLConfig

from rumor_supervisor import BaselineSupervisor
from rumor_supervisor.envcitizen import TrackTwoEnvCitizen
from rumor_supervisor.rumor_spreader import RumorSpreader
from rumor_supervisor.supervisor import SupervisorConfig
from rumor_supervisor.workflows import TRACK_TWO_EXPERIMENT

MY_PARAMS = SupervisorConfig(
    # change the parameters here, FOR FRONTEND DESIGNERS
)


async def main():
    llm_configs = [
        LLMConfig(
            provider=LLMProviderType.Qwen,
            base_url=None,
            api_key="YOUR_API_KEY",
            model="CHOOSE_YOUR_MODEL",
            semaphore=200,
        ),
        # You can add more LLMConfig here
    ]
    env_config = EnvConfig(
        pgsql=PostgreSQLConfig(
            enabled=False,  # Set to True if you want to visualize the simulation in the web UI. Notice that this will SLOW DOWN the simulation.
            dsn="YOUR_POSTGRESQL_DSN",
            num_workers="auto",
        ),
        mlflow=MlflowConfig(  # Do not need mlflow in this simulation
            enabled=False,
            mlflow_uri="http://localhost:5000",
            username="admin",
            password="admin",
        ),
        avro=AvroConfig(
            enabled=True,
        ),
    )
    map_config = MapConfig(
        file_path="THE_PATH_TO_YOUR_MAP_FILE",
        cache_path="THE_PATH_TO_YOUR_MAP_CACHE_FILE",
    )

    config = Config(
        llm=llm_configs,
        env=env_config,
        map=map_config,
        agents=AgentsConfig(
            citizens=[
                AgentConfig(
                    agent_class=TrackTwoEnvCitizen,
                    memory_from_file="./data/population_profiles.json",
                ),
                AgentConfig(
                    agent_class=RumorSpreader,
                    memory_from_file="./data/spreader_profile.json",
                ),
            ],
            supervisor=AgentConfig(
                agent_class=BaselineSupervisor,
                memory_from_file="./data/supervisor_profile.json",
            ),
        ),  # type: ignore
        exp=TRACK_TWO_EXPERIMENT,
    )  # type: ignore
    agentsociety = AgentSociety(config, tenant_id="DEMO")
    await agentsociety.init()
    await agentsociety.run()
    try:
        await agentsociety.close()
    except Exception as e:
        pass
    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
