# dl_train_mlp.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from postprocess.featurizer_dl import load_dataset, resample_traj, normalize_traj, compute_features
from postprocess.augment_dl import compose_augment

# ---------- 데이터 로드 ----------

def build_feature_dataset(data_root, newdata_root=None, target_len=100,
                          augment_cfg=None, augment_factor=0):
    # 기본: 원본
    X, y, label_names = load_dataset(data_root, target_len=target_len)

    # newdata도 합치고 싶으면 여기서 합침
    if newdata_root is not None:
        X2, y2, _ = load_dataset(newdata_root, target_len=target_len)
        X = np.concatenate([X, X2], axis=0)
        y = np.concatenate([y, y2], axis=0)

    # 원본 궤적 파일 단위 증강 루프를 별도로 돌고 싶으면, traj-level 로더를 따로 두고 사용
    # 여기서는 예시로 feature space에서 augmentation 없이, 원본 좌표에 augmentation 걸고 다시 피처 추출 [file:query]
    if augment_cfg is not None and augment_factor > 0:
        # 단, 여기에는 raw traj가 없으므로 실제 프로젝트에서는
        # "load_trajectory_file"을 써서 traj 리스트를 별도로 관리하는 것이 좋음.
        # 개념적으로는:
        pass

    return X, y, label_names

# ---------- 모델 정의 ----------

def build_mlp(input_dim, num_classes):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, kernel_initializer='he_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    data_root = 'data'       # 원본 [file:1]
    newdata_root = 'newdata' # 노이즈 데이터 [file:query]

    X, y, label_names = build_feature_dataset(data_root, newdata_root)
    print("X shape:", X.shape, "classes:", label_names)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = build_mlp(input_dim=X.shape[1], num_classes=len(label_names))
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=10, min_lr=1e-5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=cb,
        verbose=2
    )

    # 최종 평가
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}")

    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    print(classification_report(y_val, y_val_pred, target_names=label_names))

    # 모델 / 스케일러 저장(Optional)
    model.save('models/mlp_traj.h5')
    # scaler는 joblib 등으로 저장

if __name__ == "__main__":
    main()
