IP=localhost
PORT=12244

python scripts/clone.py \
    --ip $IP --port $PORT \
    --speakers_path ./assets/speakers \
    --loudness 30.0 \
    --base64
