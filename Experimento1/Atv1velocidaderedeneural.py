import numpy as np
import scipy.signal
import time
import matplotlib.pyplot as plt

def teste_convolucao_1d():
    print("=== EXPERIMENTO 1: Comparativo Matemático (Sinais 1D) ===")

    # 1. Configuração (Sinais grandes para evidenciar a diferença)
    # Tente aumentar esses números se seu PC for muito rápido
    tamanho_sinal = 200000  # 200 mil amostras
    tamanho_filtro = 20000  # 20 mil amostras no filtro

    print(f"Gerando vetores: Sinal={tamanho_sinal}, Filtro={tamanho_filtro}")
    sinal = np.random.rand(tamanho_sinal)
    filtro = np.random.rand(tamanho_filtro)

    # 2. Método Direto (Força Bruta - Similar ao deslizamento espacial)
    print("\nExecutando Convolução Direta (aguarde, pode demorar)...")
    inicio = time.time()
    # method='direct' força o cálculo O(N^2)
    scipy.signal.convolve(sinal, filtro, method='direct')
    fim = time.time()
    tempo_direto = fim - inicio
    print(f"--> Tempo Método Direto: {tempo_direto:.4f} segundos")

    # 3. Método via FFT (Teorema da Convolução)
    print("\nExecutando Convolução via FFT...")
    inicio = time.time()
    # method='fft' usa FFT -> Multiplicação -> IFFT
    scipy.signal.convolve(sinal, filtro, method='fft')
    fim = time.time()
    tempo_fft = fim - inicio
    print(f"--> Tempo Método FFT:    {tempo_fft:.4f} segundos")

    # 4. Resultados
    aceleracao = tempo_direto / tempo_fft
    print(f"\nRESULTADO: A FFT foi {aceleracao:.2f} vezes mais rápida.")

    # Gráfico simples para o relatório
    metodos = ['Direto (Espacial)', 'FFT (Frequência)']
    tempos = [tempo_direto, tempo_fft]

    plt.figure(figsize=(8, 5))
    barras = plt.bar(metodos, tempos, color=['red', 'green'])
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Impacto da FFT no Tempo de Processamento')

    # Adiciona o valor em cima da barra
    for rect in barras:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.4f}s', ha='center', va='bottom')

    plt.show()

if __name__ == "__main__":
    teste_convolucao_1d()
