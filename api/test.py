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


BASE_URL = "127.0.0.1:12244"

SAVE_GENERATED_AUDIO = False  ## control whether to save generated audios
WHETHER_TO_TEST_MTTFF = True  ## control whether to test mttff metric

SAVED_ROOT = "./results"
WARMUP_MAX_NUM_REQUESTS = 1
WARMUP_TIMES = 1
MAX_NUM_REQUESTS = 20
TEST_TIMES = 1

DOCUMENTS = [
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
    sample_rate = 24000

    async with connect(
        f"ws://{BASE_URL}/tts",
        open_timeout=120,
        ping_timeout=120,
        close_timeout=120,
    ) as websocket:
        # send a request
        await websocket.send(
            json.dumps(
                {
                    "req_params": {
                        "prompt_id": prompt_id,
                        "audio_format": "wav",
                        "sample_rate": sample_rate,
                        "instruct_text": None,
                        # "slice_seconds": 0.1
                    }
                }
            )
        )

        async def text_sender():
            async for t in random_segment_text(text):
                await websocket.send(json.dumps({"text": t, "done": False}))
                await asyncio.sleep(1e-7)
            await websocket.send(json.dumps({"text": "", "done": True}))

        asyncio.create_task(text_sender())

        all_recv_seconds = []
        whole_audio = []
        root = None
        whole_start = time.perf_counter()
        while True:
            frame_start = time.perf_counter()
            message = await websocket.recv(False)
            message = json.loads(message)

            if message["error"] or message["is_end"]:
                if message["error"]:
                    print(f"{task_id:02} ERROR:", message)

                whole_seconds = time.perf_counter() - whole_start
                all_recv_seconds.append(whole_seconds)
                print(f"{task_id:02} TOTAL RECV:", whole_seconds)
                break

            else:
                frame_seconds = time.perf_counter() - frame_start
                all_recv_seconds.append(frame_seconds)
                print(f"{task_id:02}-{message['index']:02} FRAME RECV:", frame_seconds)

                chunk = any_format_to_ndarray(message["data"], message["audio_format"], message["sample_rate"], sample_rate)
                whole_audio.append(chunk)
                if SAVE_GENERATED_AUDIO:
                    root = f"{saved_dir}/{task_id}_{prompt_id}_{message['id']}"
                    os.makedirs(f"{root}/chunks", exist_ok=True)
                    save_audio(chunk, f"{root}/chunks/{message['index']}.wav", sample_rate)

        whole_audio = np.concatenate(whole_audio)
        duration = len(whole_audio) / sample_rate

        if SAVE_GENERATED_AUDIO:
            save_audio(whole_audio, f"{root}/whole.wav", sample_rate)

        return round(all_recv_seconds[0], 4), round(all_recv_seconds[-1] / duration, 4)


def run_tts(*args):
    return asyncio.run(request_tts(*args))


def eval(num_requests=1, test_times=4, saved_root="./results", verbose=True):
    if verbose:
        print(f"========== EVAL [{num_requests} REQUESTS]  ==========")
    saved_dir = os.path.join(
        saved_root,
        "audios",
        f"{str(time.time()).split('.')[0]}_{uuid.uuid4().hex[:7]}",
    )

    ttffs = []
    rtfs = []
    for i in range(test_times):
        if verbose:
            print(f"========== {i + 1}/{test_times} ==========")
        tasks = []
        with ProcessPoolExecutor(max_workers=num_requests) as pool:
            for j in range(num_requests):
                tasks.append(
                    pool.submit(
                        run_tts,
                        random.choice(DOCUMENTS),
                        random.choice(PROMPT_IDS),
                        i + j,
                        saved_dir,
                    )
                )
        for t in tasks:
            ttff, rtf = t.result()
            ttffs.append(ttff)
            rtfs.append(rtf)

    mean_ttff = sum(ttffs) / len(ttffs)
    mean_rtf = sum(rtfs) / len(rtfs)
    if verbose:
        print(f"--> {mean_ttff=} {mean_rtf=}")
    return mean_ttff, mean_rtf


if __name__ == "__main__":
    # check speakers
    existed_speakers = requests.get(f"http://{BASE_URL}/speakers").json()
    print(f"Existed Speakers: {existed_speakers}")
    for id in list(PROMPT_IDS):
        if id not in existed_speakers:
            PROMPT_IDS.remove(id)

    ## warm up inference
    print("========== WARM UP ==========")
    eval(WARMUP_MAX_NUM_REQUESTS, WARMUP_TIMES, SAVED_ROOT, verbose=False)

    if not WHETHER_TO_TEST_MTTFF:
        exit()

    ## eval mttff (mean time to first frame) and rtf
    start = time.perf_counter()
    all_results = []
    try:
        for i in range(1, MAX_NUM_REQUESTS + 1):
            all_results.append((i, *eval(i, TEST_TIMES, SAVED_ROOT)))
    finally:
        test_time = time.perf_counter() - start
        print(f"========== TOTAL TEST TIME: {test_time} ==========")

        ## save result
        if len(all_results) > 0:
            xs = [i[0] for i in all_results]
            ys_mttff = [i[1] for i in all_results]
            ys_mrtf = [i[2] for i in all_results]

            saved_dir = os.path.join(SAVED_ROOT, "eval_results")
            os.makedirs(saved_dir, exist_ok=True)
            fig_path = os.path.join(saved_dir, f"{str(time.time()).split('.')[0]}_{uuid.uuid4().hex[:7]}.png")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            bar1 = ax1.bar(xs, ys_mttff, align="center", color="skyblue")
            ax1.set_xticks(xs)
            ax1.bar_label(bar1)
            ax1.set_title(f"Mean TTFF (test time: {test_time:.2f}s)")
            ax1.set_xlabel("num_requests")
            ax1.set_ylabel("seconds")

            bar2 = ax2.bar(xs, ys_mrtf, align="center", color="salmon")
            ax2.set_xticks(xs)
            ax2.bar_label(bar2)
            ax2.set_title("Mean RTF")
            ax2.set_xlabel("num_requests")
            ax2.set_ylabel("ratio")

            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
