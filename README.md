# MRDT

여러 시점에서 찍은 물체 이미지와 원하는 방향을 입력받아 **256×256 RGB 이미지**를 생성하는 연구 코드입니다. 현재 MVImgNet2.0 representation learning 버전을 공유합니다.

**입력 → 출력:** context 1–5장 + 각 이미지/target의 연속 방위각·고도각(sin/cos 4D) → 고정 SigLIP → RDT 생성기 → 고정 VAE decoder → target 이미지. 학습에서는 context 2–5장을 사용합니다. Mask·roll 정렬·중앙 배치·크기 정규화는 이미지 전처리에서 수행합니다. 생성기는 약 **161M parameters**, scratch 초기화이며 현재 loss는 **clean latent MSE (`prediction_type=sample`)**입니다.

## 1. 설치

현재 검증 환경: **Linux, Python 3.10, CUDA PyTorch 2.1.0, NVIDIA GPU 2개**. 학습 기본값은 GPU당 batch 16, global 32, fp32입니다.

```bash
git clone https://github.com/LimHaksoo/MRDT.git
cd MRDT
conda create -n mrdt python=3.10 -y
conda activate mrdt
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-mrdt.txt
```

## 2. 데이터·모델 준비

데이터와 checkpoint는 별도로 준비해야 합니다. MVImgNet2.0의 RGB·mask·COLMAP pose 및 **검증된 전체 catalog**를 사용합니다. 현재 확보 범위는 41개 archive, 309개 class, train 165,710 / val 9,310 / test 9,258개 객체입니다. Dataset 전체 공식 규모와 구분합니다.

```bash
# 준비된 데이터 경로로 수정: catalog.sqlite, readiness.json, preparation_config.json 필요
export DATA=/path/to/processed_plan_b_full/bin15_masked_catalog_v2
export SIGLIP=/path/to/models/siglip-so400m-patch14-384
export VAE=/path/to/models/sd-vae-ft-mse
export RUN="$PWD/runs/continuous_scratch"

huggingface-cli download google/siglip-so400m-patch14-384 --local-dir "$SIGLIP"
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "$VAE"
```

Catalog에는 원본 파일의 절대경로가 포함됩니다. 다른 서버로 옮길 때는 데이터 경로·검증 기록을 함께 준비해야 합니다. 현재 full trainer는 41개 archive의 준비 완료 상태를 검사합니다. 자세한 구성은 [데이터 안내](docs/DATA.md)를 참고하세요.

## 3. 처음부터 학습

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 mrdt_train.py \
  --data-root "$DATA" --output "$RUN" --scratch --steps 100000 \
  --siglip-model "$SIGLIP" --vae-model "$VAE" --memory-cap-gib 14
```

`--scratch`는 생성기/optimizer를 새로 초기화합니다. SigLIP과 VAE는 사전학습 가중치를 고정해서 사용합니다. 15° bin은 시점 선택용이며 **모델 조건은 양자화하지 않은 연속 각도**입니다. VRAM cap은 PyTorch allocator 상한이며 프로세스 전체 메모리의 절대 상한은 아닙니다.

## 4. 중단·재개와 결과 위치

```bash
# 안전한 중단: 현재 update 완료 후 model/optimizer/step/rank RNG 저장
touch "$RUN/STOP"

# training_status.json의 stopped 상태와 프로세스 종료를 확인한 뒤 재개
rm "$RUN/STOP"
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 mrdt_train.py \
  --data-root "$DATA" --output "$RUN" --steps 100000 \
  --resume "$RUN/checkpoints/latest.pt" \
  --siglip-model "$SIGLIP" --vae-model "$VAE" --memory-cap-gib 14
```

학습·모델 경로·소스·데이터와 총 `--steps`는 checkpoint 설정과 같아야 합니다. `--steps`는 누적 목표 step입니다. 환경이 다른 checkpoint를 재개하기 위해 검증 assertion을 지우지 마세요.

- `checkpoints/latest.pt`: 1,000 step마다 및 정상 중단 시 저장.
- `checkpoints/step_*.pt`: 5,000 step마다 및 마지막 step 저장.
- `samples/`: 1,000 step마다 기본 train/val 패널, 5,000 step마다 추가 고정 val 패널.
- `training_status.json`, `config_used.json`: 진행 상황과 실제 설정.

## 5. 원하는 방향으로 생성

```bash
CUDA_VISIBLE_DEVICES=0 python mrdt_sample.py \
  --checkpoint "$RUN/checkpoints/latest.pt" \
  --catalog "$DATA/catalog.sqlite" --split val --index 0 \
  --azimuth 23.25 --elevation -12.4 \
  --siglip-model "$SIGLIP" --vae-model "$VAE" \
  --output outputs/view.png
```

선택한 객체의 context만 읽고, target RGB·mask는 읽지 않습니다. 각도는 객체별 reference basis 기준입니다. `view.png`, 사용한 `view_context*.png`, 각도/시점 기록 `view.json`을 저장합니다. `--context-ids 1 3 7`로 catalog image ID를 지정할 수 있습니다. `--prepare-only`를 추가하면 checkpoint/GPU 없이 전처리 입력만 확인합니다.

## 코드 구성·출처

`plan_b/mvimgnet/`은 현재 dataset·전처리·학습, `models/`는 RDT, `rescene/`는 encoder·VAE 및 기존 재사용 모듈입니다. 내부 호환성을 위해 기존 package 이름을 유지했습니다. 과거 서버용 자동 학습/전환 controller는 포함하지 않았습니다.

[Robotics Diffusion Transformer](https://github.com/thu-ml/RoboticsDiffusionTransformer)를 기반으로 하며 원본 [라이선스](LICENSE)와 저작권 표기를 보존했습니다. [소스 출처·검증](docs/SNAPSHOT.md)을 참고하세요. 현재 생성 품질과 후속 anomaly 학습 성능은 검증 중입니다.
