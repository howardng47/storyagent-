# generate_mystery_town_dataset.py
from openai import OpenAI
import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()  # 默认会在当前目录和上级目录寻找 .env

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("没有找到 OPENAI_API_KEY，请先在系统环境变量里设置你的 API key。")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ===================== 这里换成你自己的大模型 API 调用 =====================

def call_llm(system_prompt,
             user_prompt,
             model_name="gpt-4.1-mini",
             temperature=0.8,
             max_tokens=512,
             response_format=None):
    """
    通用 LLM 调用封装：
    - 普通对话：不传 response_format
    - 需要 JSON：传 response_format={"type": "json_object"}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ===================== 游戏状态与 Agent 定义 =====================

@dataclass
class GameState:
    background: str
    roles: Dict[str, Dict[str, Any]]   # role_name -> {public_info, secret_info}
    killer_name: str
    max_rounds: int
    logs: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "...", "text": "..."}]


class GameMaster:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.system_prompt = (
            "你是一个推理游戏主持人（GM），负责设计'推理小镇'谋杀案游戏。"
            "请严格输出 JSON，键包括：background, roles, killer_name, max_rounds。"
        )

    def create_game(self) -> GameState:
        user_prompt = (
            "设计一局发生在'推理小镇'的谋杀案推理游戏，要求：\n"
            "1. 设定小镇背景和案件基本情况（写在 background 字段）。\n"
            "2. 角色包括：killer（杀手）、detective（侦探）、npc0（普通居民，可以帮任何一方或搅浑水）。\n"
            "3. 为每个角色设计 public_info（公开身份）和 secret_info（隐藏信息，例如真实行踪、目击情况等）。\n"
            "4. 指定 killer_name 为真正的凶手的角色名（如 'killer'）。\n"
            "5. 设置 max_rounds 为一个 3~6 之间的整数，表示对话轮数。\n"
            "6. 输出 JSON 对象，不要带任何解释性文字。"
        )
        resp = call_llm(
            self.system_prompt,
            user_prompt,
            model_name=self.model_name,
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        print("GM raw response:", resp)  # 调试用

        game_json = json.loads(resp)

        return GameState(
            background=game_json["background"],
            roles=game_json["roles"],
            killer_name=game_json["killer_name"],
            max_rounds=int(game_json["max_rounds"]),
        )


class BaseAgent:
    def __init__(self, name: str, role_desc: str, model_name: str = "gpt-4o-mini"):
        self.name = name
        self.role_desc = role_desc
        self.model_name = model_name

    def act(self, game_state: GameState) -> str:
        raise NotImplementedError


class KillerAgent(BaseAgent):
    def act(self, game_state: GameState) -> str:
        role_info = game_state.roles[self.name]
        system_prompt = (
            f"你是推理小镇中的杀手。你的公开信息是：{role_info['public_info']}。"
            f"你的隐藏信息是：{role_info['secret_info']}。"
            "你的目标是：在不暴露自己真实身份的情况下，误导侦探，把怀疑引向其他人。"
            "你每次只能说一小段话（不超过80字），语气自然，不要直接承认自己是凶手。"
        )
        # 将已有对话拼成上下文
        history = "\n".join([f"{log['role']}: {log['text']}" for log in game_state.logs])
        user_prompt = (
            f"以下是目前的案情背景和所有人的发言：\n"
            f"【背景】{game_state.background}\n"
            f"【对话记录】\n{history}\n"
            "现在请你发表一段话。"
        )
        return call_llm(system_prompt, user_prompt, model_name=self.model_name, temperature=0.9, max_tokens=120)


class DetectiveAgent(BaseAgent):
    def act(self, game_state: GameState) -> str:
        role_info = game_state.roles[self.name]
        system_prompt = (
            f"你是推理小镇中的侦探。你的公开信息是：{role_info['public_info']}。"
            f"你的隐藏信息是：{role_info['secret_info']}。"
            "你的目标是：尽可能找出谁是凶手。你每次可以结合已有线索，提出问题或者发表推理。"
            "每次发言不超过100字。"
        )
        history = "\n".join([f"{log['role']}: {log['text']}" for log in game_state.logs])
        user_prompt = (
            f"以下是目前的案情背景和所有人的发言：\n"
            f"【背景】{game_state.background}\n"
            f"【对话记录】\n{history}\n"
            "现在请你结合线索，发表一段话，可以是新的怀疑、问题或阶段性推理。"
        )
        return call_llm(system_prompt, user_prompt, model_name=self.model_name, temperature=0.7, max_tokens=150)

    def final_guess(self, game_state: GameState) -> str:
        """最后一轮：给出凶手猜测（只输出角色名即可）"""
        history = "\n".join([f"{log['role']}: {log['text']}" for log in game_state.logs])
        system_prompt = (
            "你是推理小镇中的侦探，现在游戏即将结束。"
            "请根据对话记录判断谁是凶手，"
            "只输出一个角色名（例如 'killer'、'npc0' 等），不要输出其他任何文字。"
        )
        user_prompt = (
            f"【背景】{game_state.background}\n"
            f"【对话记录】\n{history}\n"
            "请只给出你认为的凶手的角色名。"
        )
        guess = call_llm(system_prompt, user_prompt, model_name=self.model_name, temperature=0.0, max_tokens=10)
        return guess.strip()


class NPCAgent(BaseAgent):
    def __init__(self, name: str, role_desc: str, stand_for: str = "chaos",
                 model_name: str = "gpt-4o-mini"):
        super().__init__(name, role_desc, model_name)
        self.stand_for = stand_for  # "killer" / "detective" / "chaos"

    def act(self, game_state: GameState) -> str:
        role_info = game_state.roles[self.name]
        base = (
            f"你是推理小镇中的普通居民（{self.name}）。"
            f"你的公开信息是：{role_info['public_info']}。"
            f"你的隐藏信息是：{role_info['secret_info']}。"
            "你可以发表证词、传闻或者主观判断。"
            "每次发言不超过80字。"
        )
        if self.stand_for == "killer":
            bias = "你内心比较偏向杀手，希望在不明显的情况下帮杀手转移怀疑。"
        elif self.stand_for == "detective":
            bias = "你内心比较偏向侦探，希望在不明显的情况下提供对侦探有利的线索。"
        else:
            bias = "你的立场混乱，可能说真话，也可能瞎编，甚至自相矛盾。"

        system_prompt = base + bias

        history = "\n".join([f"{log['role']}: {log['text']}" for log in game_state.logs])
        user_prompt = (
            f"【背景】{game_state.background}\n"
            f"【对话记录】\n{history}\n"
            "现在轮到你说话，请发表一段话。"
        )
        return call_llm(system_prompt, user_prompt, model_name=self.model_name, temperature=0.9, max_tokens=120)


# ===================== 运行多局游戏并生成数据集 =====================

def simulate_one_game(gm_model: str = "gpt-4o-mini") -> Dict[str, Any]:
    gm = GameMaster(model_name=gm_model)
    game_state = gm.create_game()

    killer_agent = KillerAgent(name="killer", role_desc="杀手", model_name=gm_model)
    detective_agent = DetectiveAgent(name="detective", role_desc="侦探", model_name=gm_model)
    npc_agent = NPCAgent(name="npc0", role_desc="居民", stand_for=random.choice(["killer", "detective", "chaos"]),
                         model_name=gm_model)

    # 简单轮流发言：每轮 K -> N -> D
    for round_id in range(game_state.max_rounds):
        for agent, name in [(killer_agent, "killer"), (npc_agent, "npc0"), (detective_agent, "detective")]:
            text = agent.act(game_state)
            game_state.logs.append({"role": name, "text": text})

    # 游戏结束，让侦探给出猜测
    detective_guess = detective_agent.final_guess(game_state)

    # 构造给小模型训练用的 prompt 和 answer
    dialogue_text = "\n".join([f"{log['role']}: {log['text']}" for log in game_state.logs])
    prompt = (
        f"背景：{game_state.background}\n\n"
        f"以下是推理小镇中各个角色围绕案件的对话记录：\n{dialogue_text}\n\n"
        "问题：根据以上对话，谁是凶手？"
        "请只回答角色名（例如 'killer'、'detective'、'npc0' 之一）。"
    )
    answer = game_state.killer_name  # 官方真相中的凶手

    return {
        "background": game_state.background,
        "logs": game_state.logs,
        "killer_name": game_state.killer_name,
        "detective_guess": detective_guess,
        "prompt": prompt,
        "answer": answer,
    }


def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/mystery_town_qwen_train.jsonl"
    num_games = 50  # 你可以先从 10~50 局开始，跑通流程再加

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_games):
            print(f"模拟第 {i+1}/{num_games} 局游戏...")
            game_data = simulate_one_game()
            f.write(json.dumps(game_data, ensure_ascii=False) + "\n")

    print(f"已写入数据集到 {out_path}")


if __name__ == "__main__":
    main()
