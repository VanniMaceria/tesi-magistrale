import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURAZIONE ---
ROOT_PATH = r"C:\Users\Lenovo\Desktop\Magistrale\Tesi\progetto\simulation_results\FEMNIST\non-iid"
TECNICHE = ["baseline", "distillation", "ordered_dropout", "quantization"]
# Cartella di output richiesta
OUTPUT_ROOT = os.path.join(ROOT_PATH, "plots", "box_plots")

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def load_final_data(entity_name):
    data = []
    for t in TECNICHE:
        sub_path = "num_clients_10" if "Client" in entity_name else ""
        filename = f"{entity_name.lower()}.csv" if "Client" in entity_name else "server_aggregate.csv"
        path = os.path.join(ROOT_PATH, t, sub_path, filename)
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Analisi sul round finale per il confronto di efficacia a regime
            last_round = df['round'].max()
            df_final = df[df['round'] == last_round].copy()
            df_final['Tecnica'] = t
            data.append(df_final)
    return pd.concat(data) if data else None

def generate_box_plots(df, entity_name):
    entity_dir = os.path.join(OUTPUT_ROOT, entity_name)
    os.makedirs(entity_dir, exist_ok=True)
    
    metriche = ['accuracy', 'loss', 'energia(J)', 'banda(MB)', 'flops_inferenza']
    
    for m in metriche:
        if m not in df.columns: continue
        plt.figure(figsize=(10, 6))
        
        # MODIFICA QUI: Aggiunto hue='Tecnica' e legend=False
        sns.boxplot(data=df, x='Tecnica', y=m, hue='Tecnica', palette="Set2", legend=False)
        
        # Anche lo swarmplot richiede hue per coerenza (opzionale ma consigliato)
        sns.swarmplot(data=df, x='Tecnica', y=m, color="black", alpha=0.5)
        
        plt.title(f"Confronto Qualità Finale {m.upper()} - {entity_name}")
        plt.xlabel("Tecnica")
        plt.ylabel(m)
        plt.tight_layout()
        
        plt.savefig(os.path.join(entity_dir, f"box_compare_{m.split('(')[0]}.png"))
        plt.close()

def main():
    entities = ["Server_Global"] + [f"Client_{i}" for i in range(10)]
    for entity in entities:
        print(f"Generando Box Plots in plots/box_plots/{entity}...")
        df = load_final_data(entity)
        if df is not None:
            generate_box_plots(df, entity)

if __name__ == "__main__":
    main()