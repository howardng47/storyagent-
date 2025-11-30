import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import requests
# ========= 环境变量 & OpenAI 客户端 =========

# 从 .env 文件加载环境变量
load_dotenv()  # 默认会在当前目录和上级目录寻找 .env

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("没有找到 OPENAI_API_KEY，请先在 .env 或系统环境变量里设置你的 API key。")

client = OpenAI(api_key=api_key)


def call_llm(system_prompt,
             user_prompt,
             model_name="gpt-4.1-mini",
             temperature=0.8,
             max_tokens=512,
             response_format=None):

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

def serpapi_search_snippets(query: str, num_results: int = 3) -> str:
    """
    使用 SerpAPI 调用 Google 搜索，返回若干条精简的搜索摘要，
    用于增强背景和角色发言。
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        # 如果没配置 key，就直接返回空字符串，不影响主流程
        return ""

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": "zh-cn",
        "num": num_results,
    }

    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("SerpAPI 搜索调用失败：", e)
        return ""

    snippets = []
    # SerpAPI 把普通结果放在 organic_results 里
    for item in data.get("organic_results", [])[:num_results]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        if not (title or snippet):
            continue
        snippets.append(f"标题：{title}\n摘要：{snippet}\n链接：{link}")

    return "\n\n".join(snippets)


def safe_json_loads(resp_text: str):
    """
    尽量从模型输出中提取 JSON：
    1. 先直接 json.loads
    2. 不行的话，截取第一个 '{' 到最后一个 '}' 再试
    3. 还不行就打印原始内容并抛错
    """
    if resp_text is None:
        raise ValueError("GM 返回为空（None），无法解析 JSON。")

    text = resp_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取大括号中的部分
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        print("=== GM raw response cannot be parsed as JSON ===")
        print(repr(text))
        print("===============================================")
        raise ValueError("无法从 GM 输出中解析出合法 JSON。")


# ===================== 游戏状态与 Agent 定义 =====================

@dataclass
class GameState:
    background: str
    roles: Dict[str, Dict[str, Any]]   # role_name -> {name, public_info, secret_info}
    killer_name: str
    max_rounds: int
    logs: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "...", "text": "..."}]
    search_context: str = ""

def get_char_name(game_state: GameState, role_id: str) -> str:
    """根据角色ID拿中文名字，拿不到就退回ID本身。"""
    role_info = game_state.roles.get(role_id, {})
    return role_info.get("name", role_id)


def build_history_text(game_state: GameState) -> str:
    """把 logs 里的 ID 全部换成中文姓名，生成给模型看的对话历史。"""
    lines = []
    for log in game_state.logs:
        role_id = log["role"]
        char_name = get_char_name(game_state, role_id)
        lines.append(f"{char_name}: {log['text']}")
    return "\n".join(lines)


class GameMaster:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.system_prompt = (
            "你是一个推理游戏主持人（GM），负责设计“推理小镇”的谋杀案游戏。\n"
            "你必须只输出一个 JSON 对象，不要输出任何解释、注释、自然语言、代码块标记或其他内容。\n"
            "JSON 顶层键必须包括：background, roles, killer_name, max_rounds。\n"
            "roles 是一个对象，键为 'killer'、'detective'、'npc0'、'npc1'、'npc2'、'npc3'，"
            "每个键对应一个对象，该对象中必须包含：\n"
            "  - name: 角色的中文姓名（例如“李华”“王敏”），不要和死者同名；\n"
            "  - public_info: 公开身份（职业、日常形象等）；\n"
            "  - secret_info: 隐藏的信息（真实动机、案发当晚行踪、隐瞒的细节等）。\n"
            "killer_name 字段必须是上述 roles 中的一个键名，表示真正的凶手是谁。\n"
            "background 需要简要交代：死者是谁（不能是 killer）、案发地点、时间和基本矛盾冲突。\n"
            "max_rounds 为 3~6 之间的整数，表示本局对话轮数。\n"
        )

    def create_game(self) -> GameState:
        search_snippets = serpapi_search_snippets(
            "小镇 谋杀案 典型动机 人物关系 小镇生活 证词",
            num_results=3
        )

        user_prompt = (
            "请根据上述要求，设计一局发生在“推理小镇”的谋杀案推理游戏。\n"
            "你可以参考以下来自真实世界的背景资料（不要照搬原文，可以改写、融合）：\n"
            "【网络搜索参考信息】：\n"
            f"{search_snippets}\n\n"
            "严格按照指定的 JSON 结构输出，不要添加多余字段。"
        )

        resp = call_llm(
            self.system_prompt,
            user_prompt,
            model_name=self.model_name,
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"},  # 强制 JSON
        )

        game_json = safe_json_loads(resp)

        return GameState(
            background=game_json["background"],
            roles=game_json["roles"],
            killer_name=game_json["killer_name"],
            max_rounds=int(game_json["max_rounds"]),
            search_context=search_snippets,
        )


class BaseAgent:
    def __init__(self, name: str, role_desc: str, model_name: str = "gpt-4.1-mini"):
        self.name = name
        self.role_desc = role_desc
        self.model_name = model_name

    def act(self, game_state: GameState) -> str:
        raise NotImplementedError


class KillerAgent(BaseAgent):
    def act(self, game_state: GameState) -> str:
        role_info = game_state.roles[self.name]
        char_name = role_info.get("name", "某人")
        system_prompt = (
            "你是推理小镇中的一名嫌疑人，系统身份：凶手。\n"
            "你的真实姓名是：%s。\n"
            "你的公开信息是：%s。\n"
            "你的隐藏信息（不要直接说出来）：%s。\n"
            "你的目标是：在不暴露自己真实身份的情况下，误导侦探，把怀疑引向其他人。\n"
            "你每次只能说一小段话（不超过80字），语气自然，不要直接承认自己是凶手。"
            % (char_name, role_info["public_info"], role_info["secret_info"])
        )

        history = build_history_text(game_state)

        user_prompt = (
            f"【案情背景】\n{game_state.background}\n\n"
            "【现实世界参考信息】（可以用来增加生活细节和职业细节，不要直接照抄）：\n"
            f"{game_state.search_context}\n\n"
            "【当前对话记录】\n"
            f"{history}\n\n"
            f"现在轮到你（{char_name}）发言，请以“{char_name}”的口吻说话，"
            "结合自身身份与可能的生活细节，自然地说出一段话。"
        )

        return call_llm(system_prompt, user_prompt,
                        model_name=self.model_name,
                        temperature=0.9, max_tokens=120)

class DetectiveAgent(BaseAgent):
    def act(self, game_state: GameState) -> str:
        role_info = game_state.roles[self.name]
        char_name = role_info.get("name", "侦探")
        system_prompt = (
            "你是推理小镇中的侦探，真实姓名：%s。\n"
            "你的公开信息是：%s。\n"
            "你的隐藏信息（不要直接说出来）：%s。\n"
            "你的目标是：尽可能找出谁是凶手，指出别人证词中的矛盾之处。\n"
            "每次发言不超过100字，可以提出尖锐问题或阶段性推理。"
            % (char_name, role_info["public_info"], role_info["secret_info"])
        )

        # 使用带中文姓名的对话历史
        history = build_history_text(game_state)

        user_prompt = (
            f"【案情背景】\n{game_state.background}\n\n"
            "【现实世界参考信息】（包括小镇生活、典型谋杀案动机等，可用作推理参考，不要直接照抄）：\n"
            f"{game_state.search_context}\n\n"
            "【当前对话记录】\n"
            f"{history}\n\n"
            f"现在轮到你（{char_name}）发言，请结合已有线索和现实常识，"
            "提出尖锐问题或阶段性推理，发言不超过100字。"
        )

        return call_llm(
            system_prompt,
            user_prompt,
            model_name=self.model_name,
            temperature=0.7,
            max_tokens=150,
        )

    def final_guess(self, game_state: GameState) -> str:
        """最后一轮：给出凶手猜测（只输出中文姓名）。"""
        history = build_history_text(game_state)

        # 构造一个角色列表，帮助模型对上“谁是谁”
        roles_str = "\n".join([
            f"{info.get('name', role_id)}（角色ID：{role_id}）"
            for role_id, info in game_state.roles.items()
        ])

        system_prompt = (
            "你是推理小镇中的侦探，现在游戏即将结束。\n"
            "下面给出每个角色的ID和对应的中文姓名。\n"
            "请根据对话记录判断谁是凶手，只输出你认为的凶手的中文姓名，"
            "不要输出角色ID或其他任何文字。"
        )

        user_prompt = (
            f"【背景】{game_state.background}\n\n"
            f"【角色列表】\n{roles_str}\n\n"
            f"【对话记录】\n{history}\n\n"
            "请只给出你认为的凶手的中文姓名。"
        )

        guess = call_llm(
            system_prompt,
            user_prompt,
            model_name=self.model_name,
            temperature=0.0,
            max_tokens=10,
        )
        return guess.strip()


class NPCAgent(BaseAgent):
    def __init__(self, name, role_desc, stand_for="chaos", model_name="gpt-4.1-mini"):
        super(NPCAgent, self).__init__(name, role_desc, model_name)
        self.stand_for = stand_for  # "killer" / "detective" / "chaos"

    def act(self, game_state: GameState) -> str:
        role_info = game_state.roles[self.name]
        char_name = role_info.get("name", "居民")

        base = (
            "你是推理小镇中的普通居民，名字叫 %s。\n"
            "你的公开信息是：%s。\n"
            "你的隐藏信息（不要直接说出来）：%s。\n"
            "你可以发表证词、传闻或者主观判断，每次发言不超过80字。"
            % (char_name, role_info["public_info"], role_info["secret_info"])
        )
        if self.stand_for == "killer":
            bias = "你内心比较偏向真正的凶手，希望在不明显的情况下帮凶手转移怀疑。\n"
        elif self.stand_for == "detective":
            bias = "你内心比较偏向侦探，希望在不明显的情况下提供对侦探有利的线索。\n"
        else:
            bias = "你的立场混乱，有时说真话，有时夸大其词，甚至自相矛盾。\n"

        system_prompt = base + bias

        # 使用带中文姓名的对话历史
        history = build_history_text(game_state)

        user_prompt = (
            f"【案情背景】\n{game_state.background}\n\n"
            "【现实世界参考信息】（可以用来增加生活细节、职业习惯等，不要直接照抄）：\n"
            f"{game_state.search_context}\n\n"
            "【当前对话记录】\n"
            f"{history}\n\n"
            f"现在轮到你（{char_name}）发言，请以符合你身份的口吻，自然地说出此刻会讲的话，"
            "可以适当加入生活细节或主观评价，但不要一次性透露所有真相。"
        )

        return call_llm(
            system_prompt,
            user_prompt,
            model_name=self.model_name,
            temperature=0.9,
            max_tokens=120,
        )


# ===================== 运行多局游戏并生成数据集 =====================

def simulate_one_game(gm_model="gpt-4.1-mini"):
    gm = GameMaster(model_name=gm_model)
    game_state = gm.create_game()

    # 主角色
    killer_agent = KillerAgent(name="killer", role_desc="杀手", model_name=gm_model)
    detective_agent = DetectiveAgent(name="detective", role_desc="侦探", model_name=gm_model)

    # 多个 NPC：遍历所有 roles 里以 "npc" 开头的键
    npc_agents: List[NPCAgent] = []
    for role_id in game_state.roles.keys():
        if role_id.startswith("npc"):
            stand_for = random.choice(["killer", "detective", "chaos"])
            npc_agents.append(
                NPCAgent(name=role_id, role_desc="居民", stand_for=stand_for, model_name=gm_model)
            )

    # ========= 开局信息输出 =========
    killer_role_info = game_state.roles[game_state.killer_name]
    killer_char_name = killer_role_info.get("name", game_state.killer_name)
    detective_role_info = game_state.roles["detective"]
    detective_char_name = detective_role_info.get("name", "侦探")

    print("\n==========================================")
    print("本局故事背景：")
    print(game_state.background)
    print("------------------------------------------")
    print("真正的凶手角色ID：", game_state.killer_name)
    print("真正的凶手姓名：", killer_char_name)
    print("侦探姓名：", detective_char_name)
    print("本局最多对话轮数：", game_state.max_rounds)
    print("==========================================")

    # 发言顺序：杀手 -> 所有 NPC -> 侦探
    speaking_order: List[BaseAgent] = [killer_agent] + npc_agents + [detective_agent]

    for round_id in range(game_state.max_rounds):
        print(f"\n---------- 第 {round_id + 1} 轮对话 ----------")
        for agent in speaking_order:
            text = agent.act(game_state)
            game_state.logs.append({"role": agent.name, "text": text})

            # 拿到中文姓名
            role_info = game_state.roles[agent.name]
            char_name = role_info.get("name", agent.name)

            print(f"{char_name}：{text}")

    # 侦探最终推理
    detective_guess = detective_agent.final_guess(game_state)

    # 真实凶手姓名（根据 killer_name 这个 ID 去 roles 里查）
    true_killer_name = get_char_name(game_state, game_state.killer_name)

    print("\n========== 侦探最终推理结果 ==========")
    print("侦探认为凶手是：", detective_guess)
    print("真实凶手是：", true_killer_name)
    print("(调试) 真实凶手角色ID：", game_state.killer_name)

    print("==========================================\n")

    dialogue_text = "\n".join(["%s: %s" % (log["role"], log["text"]) for log in game_state.logs])
    prompt = (
            "背景：%s\n\n"
            "以下是推理小镇中各个角色围绕案件的对话记录（角色ID在每行开头）：\n%s\n\n"
            "问题：根据以上对话，谁是凶手？\n"
            "请只回答凶手的中文姓名。"
            % (game_state.background, dialogue_text)
    )

    return {
        "background": game_state.background,
        "roles": game_state.roles,
        "logs": game_state.logs,
        "killer_role_id": game_state.killer_name,
        "killer_name": true_killer_name,
        "detective_guess": detective_guess,
        "prompt": prompt,
        "answer": true_killer_name,  # 用人名做训练标签
    }

def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/mystery_town_qwen_train.jsonl"
    num_games = 10  # 你可以先从 10~50 局开始，跑通流程再加

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_games):
            print(f"模拟第 {i+1}/{num_games} 局游戏...")
            game_data = simulate_one_game()
            f.write(json.dumps(game_data, ensure_ascii=False) + "\n")

    print(f"已写入数据集到 {out_path}")


if __name__ == "__main__":
    main()
