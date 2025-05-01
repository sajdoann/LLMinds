# LLMinds

## RAG on AIC

To prepare an environment, create a virtual env, activate it and run:

```
pip install -r requirements.txt
```

Then, run the following:

```
python -m spacy download en_core_web_sm
```

Now you can enque jobs like the following command does:

```
sbatch submit_rag_aic.slurm