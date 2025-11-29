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
audio_path = r"musica_aqui"   
n_fft = 512
hop = 128
cmap = "twilight_shifted"
wave_color = "#4B0082"
eps = 1e-10
window_seconds = 10.0   
fps = 60                
offset_visual = 0.2
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
# Preparar Figure (Lógica Scroller / Sismógrafo)
# -----------------------
fig = plt.figure(figsize=(12, 7))
ax_wave = plt.subplot2grid((5, 1), (0, 0), rowspan=1)   
ax_spec = plt.subplot2grid((5, 1), (1, 0), rowspan=4)  

# --- Waveform ---
# O eixo X agora é relativo: vai de -10s até 0s (onde 0 é o "agora")
t_axis_wave = np.linspace(-window_seconds, 0, int(window_seconds * sr), endpoint=False)
wave_data_init = np.zeros_like(t_axis_wave)
wave_line, = ax_wave.plot(t_axis_wave, wave_data_init, linewidth=0.8, color=wave_color)

ax_wave.set_xlim(-window_seconds, 0) # Fixa o eixo no passado
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.set_ylabel("Amplitude")
ax_wave.set_xticks([])  
ax_wave.set_title("Waveform (Tempo Real)")

# --- Spectrogram ---
# Cria buffer inicial vazio (vmax/vmin definidos anteriormente)
db_buffer = np.full((db.shape[0], cols_window), vmin, dtype=np.float32)

# O extent também muda: x vai de -window_seconds até 0
im = ax_spec.imshow(db_buffer, aspect="auto", origin="lower", cmap=cmap,
                    extent=[-window_seconds, 0, 0, sr/2], vmin=vmin, vmax=vmax)

ax_spec.set_xlabel("Tempo Relativo (s)")
ax_spec.set_ylabel("Frequência (Hz)")
ax_spec.set_title("Espectrograma (Scrolling)")

# Barra de cor (Mantenha igual)
cbar = fig.colorbar(im, ax=ax_spec, orientation="vertical", pad=0.02)
cbar.set_label("Intensidade (dB)")

plt.tight_layout()

# -----------------------
# Variáveis de Estado
# -----------------------
audio_started = False
start_time = 0
warmup_counter = 0
WARMUP_FRAMES = 15  # Aquecimento para evitar travadas

# -----------------------
# Função de Atualização (Lógica Scroller)
# -----------------------
def update(frame):
    global audio_started, start_time, warmup_counter

    # 1. Aquecimento (Warmup)
    if not audio_started:
        if warmup_counter < WARMUP_FRAMES:
            warmup_counter += 1
            return im, wave_line
        
        # Inicia áudio sincronizado
        sd.stop()
        sd.play(audio, sr)
        start_time = time.time()
        audio_started = True
        return im, wave_line

    # 2. Calcula tempo decorrido
    tempo_passado = (time.time() - start_time) + offset_visual
    if tempo_passado > tempo_total:
        tempo_passado = tempo_total

    # --- ESPECTROGRAMA (Preenchimento da direita para esquerda) ---
    col_end_idx = int(round(tempo_passado * cols_per_second))
    col_start_idx = col_end_idx - cols_window

    # Buffer vazio para o frame atual
    view_data = np.full((db.shape[0], cols_window), vmin, dtype=np.float32)

    # Se já temos dados para mostrar
    if col_end_idx > 0:
        # Índices reais no array de dados completo (db)
        idx_dados_start = max(0, col_start_idx)
        idx_dados_end = min(cols_total, col_end_idx)
        
        data_slice = db[:, idx_dados_start:idx_dados_end]
        w_slice = data_slice.shape[1]
        
        # Coloca os dados na parte direita do buffer visual
        if w_slice > 0:
            view_data[:, -w_slice:] = data_slice

    im.set_data(view_data)
    # Mantém o extent fixo para não pular
    im.set_extent([-window_seconds, 0, 0, sr/2])

    # --- WAVEFORM (Preenchimento da direita para esquerda) ---
    idx_audio_end = int(tempo_passado * sr)
    idx_audio_start = idx_audio_end - int(window_seconds * sr)
    
    wave_view = np.zeros(len(t_axis_wave), dtype=np.float32)
    
    real_start = max(0, idx_audio_start)
    real_end = min(len(audio), idx_audio_end)
    
    if real_end > real_start:
        segment = audio[real_start:real_end]
        wave_view[-len(segment):] = segment

    wave_line.set_data(t_axis_wave, wave_view)

    return im, wave_line

# -----------------------
# Iniciar
# -----------------------
ani = FuncAnimation(fig, update, interval=1000/fps, blit=True)
print("Iniciando visualizador...")
plt.show()
sd.stop()
# esperar terminar o áudio antes de fechar
sd.wait()
