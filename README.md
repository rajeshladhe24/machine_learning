# Heart Disease Prediction - Vercel Project

## Project files
- `public/index.html`: responsive HTML/CSS/JavaScript frontend
- `api/index.py`: FastAPI backend
- `heart_disease_model.pkl`: trained scikit-learn pipeline
- `requirements.txt`: pinned Python dependencies
- `vercel.json`: Vercel configuration

## Deploy
1. Push the complete contents of this folder to the root of your GitHub repository.
2. Import the repository into Vercel, or push these changes to the existing connected repository.
3. Keep the root directory as `./`; do not set a custom build command or output directory.
4. Open your Vercel domain. The frontend is served at `/`.
5. API health check: `/api/health`
6. Swagger docs: `/api/docs`
7. The frontend sends predictions to `/api/predict`.

## Local API run
Install requirements and run: `uvicorn api.index:app --reload`

## Important
This is an educational demonstration only, not a medical diagnostic tool. Do not use it for clinical or healthcare decisions.
