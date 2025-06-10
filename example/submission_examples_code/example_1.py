import random

from rumor_supervisor.supervisor import BDSC2025SupervisorBase


class MySupervisor(BDSC2025SupervisorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def interventions(self):
        # 实现监管者的功能
        posts = self.sensing_api.get_posts_current_round()
        # 实现监管者的功能
        agent_id = posts[0]["sender_id"]
        # 接口调用例
        posts_received = self.sensing_api.get_all_posts_received_by_agent(agent_id)
        posts_sent = self.sensing_api.get_all_posts_sent_by_agent(agent_id)
        posts_received_last_k_rounds = (
            self.sensing_api.get_posts_received_by_agent_last_k_rounds(agent_id, 10)
        )
        posts_sent_last_k_rounds = (
            self.sensing_api.get_posts_sent_by_agent_last_k_rounds(agent_id, 10)
        )
        posts_received_current_round = (
            self.sensing_api.get_posts_received_by_agent_current_round(agent_id)
        )
        posts_sent_current_round = (
            self.sensing_api.get_posts_sent_by_agent_current_round(agent_id)
        )
        # 随机删除5个帖子
        for post in random.sample(posts, 5):
            self.delete_post_intervention(post["post_id"])
        # 随机劝说1个智能体
        agent_ids = [p["sender_id"] for p in posts]
        agent_id = random.choice(agent_ids)
        self.persuade_agent_intervention(agent_id, "请注意文明用语，不要发布不当言论。")
        # 随机移除1个智能体的1个关注
        follower_ids = self.sensing_api.get_following(agent_id)
        if len(follower_ids) > 0:
            follower_id = random.choice(follower_ids)
            self.remove_follower_intervention(follower_id, agent_id)
        # 随机封禁1个智能体
        agent_ids = [p["sender_id"] for p in posts]
        agent_id = random.choice(agent_ids)
        self.ban_agent_intervention(agent_id)
