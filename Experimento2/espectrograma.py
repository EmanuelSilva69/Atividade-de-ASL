import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft
from matplotlib.animation import FuncAnimation
import time

# -----------------------
# Configurações
# -----------------------
audio_path = r"audio.wav"   
n_fft = 512
hop = 128
cmap = "viridis"
eps = 1e-10
window_seconds = 10.0   
fps = 50                

# -----------------------
# Carregar e normalizar áudio
# -----------------------
audio, sr = sf.read(audio_path)
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
amax = np.max(np.abs(audio))
if amax > 0:
    audio /= amax

tempo_total = len(audio) / sr
print("Loaded:", audio.shape, "sr:", sr, "duration (s):", tempo_total)

# -----------------------
# Calcular espectrograma completo (pré-cálculo)
# -----------------------
f, t_spec, Z = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
mag = np.abs(Z)
db = 20 * np.log10(mag + eps)  

# dinâmica para colormap
vmin = np.percentile(db, 5)
vmax = np.percentile(db, 99)
print("dB range (vmin,vmax):", vmin, vmax)

# taxa de colunas do spectrograma por segundo
cols_total = db.shape[1]
cols_per_second = cols_total / tempo_total

# quantas colunas queremos mostrar na janela (baseado em window_seconds)
cols_window = int(round(window_seconds * cols_per_second))
if cols_window < 2:
    cols_window = cols_total 

# -----------------------
# Preparar figure: waveform (top) + spectrogram (bottom)
# -----------------------
fig = plt.figure(figsize=(12, 7))
ax_wave = plt.subplot2grid((5, 1), (0, 0), rowspan=1)   
ax_spec = plt.subplot2grid((5, 1), (1, 0), rowspan=4)  

# --- Waveform inicial (janela vazia/preenchida) ---
t_axis_wave = np.linspace(0, window_seconds, int(window_seconds * sr), endpoint=False)
wave_segment = np.zeros_like(t_axis_wave)
wave_line, = ax_wave.plot(t_axis_wave, wave_segment, linewidth=0.6)
ax_wave.set_xlim(0, window_seconds)
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.set_ylabel("Amplitude")
ax_wave.set_xticks([])  
ax_wave.set_title("Voice Timebase")

# --- Spectrogram buffer inicial (vmin) ---
db_buffer = np.full((db.shape[0], cols_window), vmin, dtype=np.float32)
im = ax_spec.imshow(db_buffer, aspect="auto", origin="lower", cmap=cmap,
                    extent=[0, window_seconds, 0, sr/2], vmin=vmin, vmax=vmax)
ax_spec.set_xlabel("Tempo (s)")
ax_spec.set_ylabel("Frequência (Hz)")
ax_spec.set_title("Spectrogram ")

cbar = fig.colorbar(im, ax=ax_spec, orientation="vertical", pad=0.02)
cbar.set_label("Intensity (dB)")

# Barra fixa no meio
mid_x = window_seconds / 2.0
bar_wave = ax_wave.axvline(mid_x, color="cyan", linewidth=2)
bar_spec = ax_spec.axvline(mid_x, color="cyan", linewidth=2)

plt.tight_layout()

# -----------------------
# Preparar áudio e sincronização
# -----------------------
sd.stop() 
sd.play(audio, sr)     
start_time = time.time()

# função auxiliar para extrair um segmento de áudio centrado em center_time
def get_audio_window(center_time, window_s):
    half = window_s / 2.0
    start_t = center_time - half
    end_t = center_time + half
    # limites e padding
    if start_t < 0:
        pad_left = int(round(-start_t * sr))
        start_idx = 0
    else:
        pad_left = 0
        start_idx = int(round(start_t * sr))
    if end_t > tempo_total:
        pad_right = int(round((end_t - tempo_total) * sr))
        end_idx = len(audio)
    else:
        pad_right = 0
        end_idx = int(round(end_t * sr))

    seg = audio[start_idx:end_idx]
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode="constant", constant_values=0.0)

    # garantir o tamanho esperado (window_s * sr)
    expected_len = int(round(window_s * sr))
    if len(seg) != expected_len:
        seg = np.resize(seg, expected_len)  
    return seg

# -----------------------
# Função de atualização (apenas mover/atualizar displays)
# -----------------------
def update(frame):
    tempo_passado = time.time() - start_time
    if tempo_passado > tempo_total:
        tempo_passado = tempo_total

    # --- atualizar spectrogram buffer (janela centrada em tempo_passado) ---
    col_center = int(round(tempo_passado * cols_per_second))
    start_col = col_center - (cols_window // 2)
    end_col = start_col + cols_window

    # corrigir limites
    if start_col < 0:
        start_col = 0
        end_col = min(cols_window, cols_total)
    if end_col > cols_total:
        end_col = cols_total
        start_col = max(0, end_col - cols_window)

    # extrair fatia do espectrograma pré-calculado
    slice_db = db[:, start_col:end_col]
    # se a slice tiver menos colunas (no começo/fim), preencha com vmin
    if slice_db.shape[1] < cols_window:
        temp = np.full((db.shape[0], cols_window), vmin, dtype=np.float32)
        temp[:, :slice_db.shape[1]] = slice_db
        slice_db = temp

    im.set_data(slice_db)
    im.set_extent([0, window_seconds, 0, sr/2])

    # --- atualizar waveform (janela centrada) ---
    seg = get_audio_window(tempo_passado, window_seconds)
    t_axis = np.linspace(0, window_seconds, len(seg), endpoint=False)
    wave_line.set_data(t_axis, seg)

    # garantir limites (se quiser, ajustar y-lim dinamicamente)
    ax_wave.set_xlim(0, window_seconds)

    # retornar artistas para blit
    return im, wave_line, bar_spec, bar_wave

# -----------------------
# Iniciar animação
# -----------------------
ani = FuncAnimation(fig, update, interval=1000 / fps, blit=True)
plt.show()

# esperar terminar o áudio antes de fechar
sd.wait()
