import requests
from utils import *
import os
import asyncio
from websockets.asyncio.client import connect
import json
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
import random
import matplotlib.pyplot as plt


BASE_URL = "localhost:12244"
SAVED_ROOT = "./results"
SAVE_GENERATED_AUDIO = True  ## control whether to save generated audios
WHETHER_TO_TEST_MTTFF = False  ## control whether to test mttff metric
TEXTS = [
    # """《404病房》
    # 护士站的值班表上并没有404号病房。但凌晨三点，我分明听见走廊尽头传来规律的滴水声。
    # 白大褂被冷汗浸透时，我握着手电推开了那扇漆皮剥落的铁门。
    # 生锈的轮椅在月光下空转，霉味里混着福尔马林之外的腥甜。镜面碎裂的洗手池滴答作响，每声都精准卡在心跳间隙。
    # 当我打开最里侧的储物柜，整排玻璃药瓶突然同时炸裂，飞溅的碎片却在半空诡异地凝滞成某种符号。
    # 镜中倒影忽然眨了眨眼——那不是我。苍白的手指从背后缠上脖颈时，我听见病历卡散落的声音。
    # 最后一张泛黄的纸片上，二十年前的潦草笔迹写着我的名字，诊断栏里爬满黑虫般的字迹：该患者坚称在404病房工作。
    # 晨光刺破窗棂时，巡逻保安在废弃仓库发现了十三支空镇静剂。监控录像里，我整夜都坐在布满灰尘的镜子前，对着空气微笑。""",
    ##############################
    # """《午夜镜像》
    # 电梯在13楼停下时，显示屏分明跳动着"12"。走廊尽头的古董穿衣镜是房东特意叮嘱不能挪动的，此刻镜面却浮着层水雾，把月光滤成尸斑似的青灰。
    # 我第37次擦去雾气时，掌纹突然在镜中扭曲成陌生纹路。镜框雕花的藤蔓缠住手腕刹那，整栋楼的声控灯同时爆裂。
    # 黑暗中，镜里的我扬起嘴角，而真正的我正死死咬住嘴唇。
    # 保安第二天发现镜子碎成蛛网状，每块碎片都映着不同时间的我：03:15惊恐后退，03:17脖颈后仰成诡异角度，03:19的碎片却空无一人。
    # 监控显示我整夜都紧贴镜面站立，指尖在玻璃上重复书写着1937年的日期。
    # 如今租客们总抱怨13楼有面擦不干净的镜子。偶尔有醉鬼看见穿真丝睡裙的女人在镜前梳头，发梢滴落的不知是水还是血——那件染血的睡裙，
    # 此刻正整整齐齐叠在我的衣柜底层。""",
    ##############################
    # """曲曲折折的荷塘上面，弥望的是田田的叶子。叶子出水很高，像亭亭的舞女的裙。
    # 层层的叶子中间，零星地点缀着些白花，有袅娜地开着的，有羞涩地打着朵儿的；
    # 正如一粒粒的明珠，又如碧天里的星星，又如刚出浴的美人。""",
    ##############################
    # """五年前那个深秋的黄昏，我独自站在地坛的古柏下，听见风穿过荒草与断墙，仿佛时光碎裂的声音。
    # 母亲曾在这里寻过我无数次，而她的脚印早已被落叶掩埋，只剩下我与这园子互相望着，像两个被遗弃的孩子。"""
    ##############################
    # """若前方无路，我便踏出一条路；若天理不容，我便逆转这乾坤。"""
    ##############################
    """今天天气真好呀！让我们一起出去玩吧。""",
    ##############################
    # ""
]
PROMPT_IDS = [
    # "roumeinvyou",
    # "sunwukong",
    # "zhenhuan",
    # "teemo",
    # "lindaiyu",
    # "linjianvhai",
    # "twitch",
    # "wenroutaozi",
    # "tianxinxiaomei",
    # "yangguangtianmei",
    "nezha",
]


async def random_segment_text(text, max_len=10):
    while len(text) > 0:
        l = random.randint(1, max_len)
        yield text[:l]
        text = text[l:]


async def request_tts(text, prompt_id, task_id, saved_dir):
    async with connect(f"ws://{BASE_URL}/tts") as websocket:
        # send a request
        await websocket.send(
            json.dumps(
                {
                    "req_params": {
                        "prompt_id": prompt_id,
                        "audio_format": "mp3",
                        "sample_rate": 24000,
                        "instruct_text": None,
                    }
                }
            )
        )

        async def send_text_stream():
            async for t in random_segment_text(text):
                await websocket.send(json.dumps({"text": t, "done": False}))
                await asyncio.sleep(1e-7)
            await websocket.send(json.dumps({"text": "", "done": True}))

        asyncio.create_task(send_text_stream())

        all_recv_seconds = []
        whole_audio = []
        root = None
        resample_rate = 24000
        whole_start = time.perf_counter()
        while True:
            frame_start = time.perf_counter()
            message = await websocket.recv(False)
            message = json.loads(message)

            if message["error"]:
                print(f"{task_id:02} ERROR:", message)

            elif not message["is_end"]:
                frame_seconds = time.perf_counter() - frame_start
                all_recv_seconds.append(frame_seconds)
                print(f"{task_id:02}-{message['index']:02} FRAME RECV:", frame_seconds)

                if SAVE_GENERATED_AUDIO:
                    chunk = any_format_to_ndarray(
                        message["data"],
                        message["audio_format"],
                        message["sample_rate"],
                        resample_rate,
                    )
                    whole_audio.append(chunk)
                    root = f"{saved_dir}/{task_id}_{prompt_id}_{message['id']}"
                    os.makedirs(f"{root}/chunks", exist_ok=True)
                    save_audio(
                        chunk,
                        f"{root}/chunks/{message['index']}.mp3",
                        resample_rate,
                    )

            elif message["is_end"]:
                whole_seconds = time.perf_counter() - whole_start
                all_recv_seconds.append(whole_seconds)
                print(f"{task_id:02} ALL RECV:", whole_seconds)
                break

        if SAVE_GENERATED_AUDIO:
            save_audio(np.concatenate(whole_audio), f"{root}/whole.mp3", resample_rate)

        return all_recv_seconds[0]


def run_tts(*args):
    return asyncio.run(request_tts(*args))


def test_mttff(num_requests=1, test_times=4, saved_root="./results"):
    print(f"========== TEST TTFF -- {num_requests} REQUESTS  ==========")
    saved_dir = os.path.join(
        saved_root,
        "speeches",
        f"{str(time.time()).split('.')[0]}_{uuid.uuid4().hex[:7]}",
    )

    mean_ttffs = []
    for i in range(test_times):
        print(f"========== {i + 1}/{test_times} ==========")
        tasks = []
        with ProcessPoolExecutor(max_workers=num_requests) as pool:
            for j in range(num_requests):
                tasks.append(
                    pool.submit(
                        run_tts,
                        random.choice(TEXTS),
                        random.choice(PROMPT_IDS),
                        i + j,
                        saved_dir,
                    )
                )
        mttff = sum([t.result() for t in tasks]) / len(tasks)
        mean_ttffs.append(mttff)

    avg_mttff = sum(mean_ttffs) / len(mean_ttffs)
    print("--> MEAN TTFT:", avg_mttff)
    return avg_mttff


if __name__ == "__main__":
    # check speakers
    existed_speakers = requests.get(f"http://{BASE_URL}/speakers").json()
    print(f"Existed Speakers: {existed_speakers}")
    for id in list(PROMPT_IDS):
        if id not in existed_speakers:
            PROMPT_IDS.remove(id)

    ## warm up inference
    test_mttff(1, 1, SAVED_ROOT)

    if not WHETHER_TO_TEST_MTTFF:
        exit()

    ## test mttff (mean time to first frame)
    start = time.perf_counter()
    all_results = []
    try:
        for i in range(1, 21):
            all_results.append((i, test_mttff(i, 2, SAVED_ROOT)))
    finally:
        test_time = time.perf_counter() - start
        print(f"========== TOTAL TEST TIME: {test_time} ==========")

        ## save result
        if len(all_results) > 0:
            saved_dir = os.path.join(SAVED_ROOT, "mttff")
            os.makedirs(saved_dir, exist_ok=True)
            fig_path = os.path.join(
                saved_dir,
                f"{str(time.time()).split('.')[0]}_{uuid.uuid4().hex[:7]}.png",
            )

            xs = [x for x, _ in all_results]
            ys = [round(y, 2) for _, y in all_results]
            plt.figure(figsize=(10, 6))
            bar = plt.bar(xs, ys, align="center")
            plt.xticks(xs)
            plt.bar_label(bar)
            plt.title(f"Mean TTFF (test time: {test_time:.2f}s)")
            plt.xlabel("num_requests")
            plt.ylabel("seconds")
            plt.savefig(fig_path)
            plt.close()
