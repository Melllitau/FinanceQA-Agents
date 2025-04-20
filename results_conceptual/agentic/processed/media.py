import pandas as pd
import os
import json

# Caminho base das pastas
base_path = os.path.dirname(__file__)
rodadas = [
  ("run1", "run1_corrected"), 
  ("run2", "run2_corrected"), 
  ("run3", "run3_corrected")
]

files = [
  ("command-r7b_agent_eco.csv", "command-r7b_output_agent.json", "command-r7b_agent.txt"),
  ("deepseek-r1_14b_agent_eco.csv", "deepseek-r1_14b_output_agent.json", "deepseek-r1_14b_agent.txt"),
  ("deepseek-r1_32b_agent_eco.csv", "deepseek-r1_32b_output_agent.json", "deepseek-r1_32b_agent.txt"),
  ("gemma3_12b_agent_eco.csv", "gemma3_12b_output_agent.json", "gemma3_12b_agent.txt"),
  ("gemma3_27b_agent_eco.csv", "gemma3_27b_output_agent.json", "gemma3_27b_agent.txt"),
  ("qwen2.5_14b_agent_eco.csv", "qwen2.5_14b_output_agent.json", "qwen2.5_14b_agent.txt"),
  ("qwen2.5_32b_agent_eco.csv", "qwen2.5_32b_output_agent.json", "qwen2.5_32b_agent.txt"),
  ("qwq_agent_eco.csv", "qwq_output_agent.json", "qwq_agent.txt")
]

# Diretório para salvar os resultados
output_dir = os.path.join(base_path, "resultados")
os.makedirs(output_dir, exist_ok=True)

# Processa cada rodada
for file_name, file_json_name, file_txt_name in files:
  durations = []
  power_consumptions = []
  co2_emissions = []
  accuracies = {}  # Lista para armazenar as acurácias
  vram_usages = []  # Lista para armazenar os valores de VRAM usados
  last_row_values = []
  for rodada, rodada_corrected in rodadas:
    # Processa o arquivo CSV
    file_path = os.path.join(base_path, rodada, file_name)
    if os.path.exists(file_path):
      # Lê o CSV
      df = pd.read_csv(file_path)
      
      # Obtém a última linha
      last_row = df.iloc[-1]
      durations.append(last_row["duration(s)"])
      power_consumptions.append(last_row["power_consumption(kWh)"])
      co2_emissions.append(last_row["CO2_emissions(kg)"])
      last_row_values.append({
        "duration(s)": last_row["duration(s)"],
        "power_consumption(kWh)": last_row["power_consumption(kWh)"],
        "CO2_emissions(kg)": last_row["CO2_emissions(kg)"]
      })
    
    # Processa o arquivo JSON
    file_json_corrected = file_json_name.replace("_agent", "")
    json_path = os.path.join(base_path, rodada_corrected, file_json_corrected)
    if os.path.exists(json_path):
      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Obtém a acurácia do resumo
        if "summary" in data and "accuracy" in data["summary"]:
          accuracies[rodada] = (data["summary"]["accuracy"])
    
    # Processa o arquivo TXT
    txt_path = os.path.join(base_path, rodada, file_txt_name)
    if os.path.exists(txt_path):
      with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Obtém a última linha que contém "Maximum GPU VRAM used"
        for line in reversed(lines):
          if "Maximum GPU VRAM used" in line:
            vram_value = float(line.split(":")[-1].strip().split()[0])
            vram_usages.append(vram_value / 1024)
            break
          else:
            print("Erro")
            break

  # Calcula as médias gerais
  overall_duration_mean = sum(durations) / (len(durations) * 60 * 60)
  overall_power_mean = sum(power_consumptions) / len(power_consumptions)
  overall_co2_mean = sum(co2_emissions) / len(co2_emissions)
  overall_accuracy_mean = sum(accuracies.values()) / len(accuracies)
  overall_vram_mean = sum(vram_usages) / len(vram_usages)
  higher_accuracy_round = max(accuracies, key=accuracies.get)
  higher_accuracy_value = accuracies[higher_accuracy_round]


  file = file_name.replace("_eco.csv", "")
  # Exibe os resultados
  print("Médias gerais (" + file + "):")
  print(f"Duração média (h): {overall_duration_mean}")
  print(f"Consumo médio de energia (kWh): {overall_power_mean}")
  print(f"Emissão média de CO2 (kg): {overall_co2_mean}")
  print(f"Acurácia média: {overall_accuracy_mean}")
  print(f"VRAM média usada (GiB): {overall_vram_mean}")
  print(f"Rodada com maior acurácia: {higher_accuracy_round} ({higher_accuracy_value})")
  print()
  
  # Salva os resultados em um CSV
  output_file = os.path.join(output_dir, f"mean_{file}.csv")
  result_df = pd.DataFrame([{
    "file_name": file.replace("_eco.csv", ""),
    "duration_mean(h)": overall_duration_mean,
    "power_consumption_mean(kWh)": overall_power_mean,
    "co2_emission_mean(kg)": overall_co2_mean,
    "accuracy_mean": overall_accuracy_mean,
    "vram_mean(GiB)": overall_vram_mean,
    "highest_accuracy_round": f"{higher_accuracy_round} ({higher_accuracy_value})"
  }])
  result_df.to_csv(output_file, index=False, encoding="utf-8")
  print(f"Resultados salvos em: {output_file}")