# Heart Disease Prediction API (Vercel)

## Project structure

```text
heart-disease-vercel/
├── api/
│   └── index.py
├── heart_disease_model.pkl
├── requirements.txt
└── vercel.json
```

## Deploy

1. Push this folder's contents to a GitHub repository.
2. In Vercel, select **Add New → Project** and import the repository.
3. Keep the project root as the repository root. No build command or output directory is needed for the FastAPI app.
4. Click **Deploy**.
5. Open `https://YOUR-PROJECT.vercel.app/docs` to test the API.

## Test request

POST `/predict` with JSON:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

The model is an educational example, not a medical diagnostic tool. Only load pickle files from trusted sources.
