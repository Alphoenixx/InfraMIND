# InfraMIND v4

> **Self-Improving Infrastructure Reasoning System**
> 
> *A full-stack, GUI-driven platform for infrastructure generation, multi-objective evaluation, and RL-based simulation/optimization.*

---

## Quickstart Guide

InfraMIND v4 consists of two main components that need to run simultaneously: the **FastAPI Backend** and the **React Vite Frontend**.

### 1. Start the Backend API (Python)
The backend powers the simulation engine, the AI code generation router, and the RL/TuRBO optimization loops.

1. Open a terminal.
2. Navigate to the root directory `InfraMIND`.
3. Activate your Python environment (if you are using one).
4. Run the FastAPI server via Uvicorn:
   ```bash
   python -m uvicorn api.main:app --reload --port 8000
   ```
*The backend will now be listening on `http://localhost:8000`.*

### 2. Start the Frontend GUI (React + Vite)
The frontend is the interactive glassmorphism dashboard.

1. Open a **new**, separate terminal window.
2. Navigate into the `frontend` folder:
   ```bash
   cd frontend
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
*The frontend will start instantly on `http://localhost:5173`.*

### 3. Access the Dashboard
Once both servers are running, open your web browser and go to:
**[http://localhost:5173](http://localhost:5173)**

---

## Project Structure

*   **/api**: The FastAPI application serving REST endpoints and WebSockets.
*   **/frontend**: The React + TypeScript user interface.
*   **/optimizer**: Core Bayesian Optimization, GP surrogates, and TuRBO logic.
*   **/simulator**: SimPy discrete-event topological engine.
*   **/workloads**: Trace generation (Steady, Diurnal, Bursty profiles).
*   **/config**: System and topology configurations (`default_config.yaml`).
