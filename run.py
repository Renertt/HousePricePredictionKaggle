import papermill as pm
import json
import os
import numpy as np

def run_benchmark():
    resultsDir = 'modelsInfo/collectedData'
    outputDir = 'modelsInfo/executedNotebooks'

    os.makedirs(resultsDir, exist_ok=True)
    os.makedirs(outputDir, exist_ok=True)

    notebooks_to_run = {
        'RandomForest': 'modelsInfo/models/housePriceAnaliseRF.ipynb',
        'XGBoost': 'modelsInfo/models/housePriceAnaliseXGB.ipynb',
        'LGBoost': 'modelsInfo/models/housePriceAnaliseLGB.ipynb'
    }
    
    for modelName, notebook_path in notebooks_to_run.items():
        print(f"Starting {modelName}")
        pm.execute_notebook(
            notebook_path,
            f'modelsInfo/executedNotebooks/output_{os.path.basename(notebook_path)}',
            cwd='.'
        )
        print(f"{modelName} finished")

    finalResults = []
    resultFiles = [
        'modelsInfo/collectedData/resultsRF.json',
        'modelsInfo/collectedData/resultsXGB.json',
        'modelsInfo/collectedData/resultsLGB.json'
                   ]
    
    for file_path in resultFiles:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                finalResults.append(json.load(f))
    
    for result in finalResults:
        modelName = result['modelName']
        mapeResult = result['mae']
        mapeResult = result['mape']
        print(f"Model: {modelName:<20} | MAPE: {mapeResult:,.2f}")

    bestModelResult = min(finalResults, key=lambda x: x['mape'])

    print("\n--- Best Model ---")
    print(f"Model: {bestModelResult['modelName']} with MAPE: {bestModelResult['mape']:,.2f}")

if __name__ == '__main__':
    run_benchmark()