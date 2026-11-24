@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

rem =========================================================================
rem  start_all.bat  -  suiiiiii / CKGAI
rem
rem  Starts all three parts of the project:
rem    1. BACKEND  FastAPI            http://localhost:8000  (docs: /docs)
rem    2. ML PLANE ml\ (torch/SciBERT) warmed in its own window, loaded by API
rem    3. FRONTEND Next.js            http://localhost:3000
rem
rem  First run : creates backend\.venv, installs every dependency
rem              (core + ml + llm + graph + dev), writes backend\.env,
rem              installs npm packages, seeds demo data.
rem  Later runs: skips anything that is already in place.
rem
rem  Switches (set before running, e.g.  set SKIP_INSTALL=1 ):
rem    SKIP_INSTALL=1   do not touch python/npm dependencies
rem    INSTALL_ML=0     do not download the ML extras (torch ~2GB)
rem    SKIP_SEED=1      do not seed development data
rem    DRY_RUN=1        do everything except opening the service windows
rem =========================================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BACKEND=%ROOT%\backend"
set "FRONTEND=%ROOT%"
set "VENV=%BACKEND%\.venv"
set "PY=%VENV%\Scripts\python.exe"

echo ==========================================================================
echo   suiiiiii / CKGAI  -  start_all
echo   project : %ROOT%
echo   frontend: http://localhost:3000
echo   backend : http://localhost:8000   (API docs: http://localhost:8000/docs)
echo   ml      : warmup window + ML plane inside the API process
echo ==========================================================================
echo.

echo(%ROOT%|find " " >nul
if not errorlevel 1 echo [WARN] Path contains spaces - move the project to a space-free path if a window fails to open.
echo.

rem --------------------------------------------------------------- 0. tools
where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found on PATH. Install Node.js 18+ first.
  goto :fail
)
where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found on PATH. Reinstall Node.js.
  goto :fail
)

rem ------------------------------------------- 1. python virtual environment
if exist "%PY%" goto :venv_ready

where python >nul 2>nul
if not errorlevel 1 (
  set "BOOT_PY=python"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "BOOT_PY=py -3"
  ) else (
    echo [ERROR] No Python found on PATH. Install Python 3.11+ first.
    goto :fail
  )
)

echo [1/6] Creating virtual environment: %VENV%
%BOOT_PY% -m venv "%VENV%"
if errorlevel 1 (
  echo [ERROR] Could not create the virtual environment.
  goto :fail
)

:venv_ready
if not exist "%PY%" (
  echo [ERROR] %PY% is missing - virtual environment is broken.
  echo         Delete the backend\.venv folder and run start_all.bat again.
  goto :fail
)
echo [1/6] Virtual environment ready: %VENV%

if "%SKIP_INSTALL%"=="1" (
  echo [2/6] SKIP_INSTALL=1 - skipping python dependency check.
  echo [3/6] SKIP_INSTALL=1 - skipping ML dependency check.
  goto :env_step
)

rem ------------------------------------------------- 2. backend core packages
echo [2/6] Checking backend core dependencies...
"%PY%" -c "import fastapi,uvicorn,sqlalchemy,pydantic_settings,structlog,httpx" 2>nul
if not errorlevel 1 (
  echo       Core packages already installed.
  goto :core_done
)

echo       Installing core backend packages (fastapi, uvicorn, sqlalchemy, ...)...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -e "%BACKEND%"
if errorlevel 1 (
  echo [ERROR] Core dependency install failed. Check the output above.
  goto :fail
)
:core_done

rem ----------------------------------------------------- 3. ML plane packages
echo [3/6] Checking ML dependencies (torch, transformers, sentence-transformers)...
"%PY%" -c "import torch,transformers,sentence_transformers,sklearn" 2>nul
if not errorlevel 1 (
  echo       ML packages already installed.
  goto :ml_done
)
if "%INSTALL_ML%"=="0" (
  echo       INSTALL_ML=0 - ML extras skipped. The API boots without them.
  set "ML_OK=0"
  goto :ml_done
)

echo       First run: installing ML + LLM + graph + dev extras.
echo       This downloads PyTorch and friends and can take a long time...
"%PY%" -m pip install -e "%BACKEND%[ml,llm,graph,dev]"
if not errorlevel 1 (
  set "ML_OK=1"
  goto :ml_done
)

echo [WARN] Full extras install failed - retrying without scispacy...
echo        (scispacy/spacy has no Python 3.13 wheels; nothing in ml\ imports it)
"%PY%" -m pip install torch torch-geometric transformers sentence-transformers datasets accelerate scikit-learn google-genai neo4j
if not errorlevel 1 (
  set "ML_OK=1"
  goto :ml_done
)
echo [WARN] ML extras could not be installed on this Python version.
echo        The backend still starts; ML features will report as unavailable.
set "ML_OK=0"
:ml_done

rem ------------------------------------------------------------- 4. backend .env
:env_step
if exist "%BACKEND%\.env" (
  echo [4/6] backend\.env already exists - left untouched.
  goto :env_done
)

echo [4/6] Creating backend\.env from .env.example
copy /y "%BACKEND%\.env.example" "%BACKEND%\.env" >nul
if errorlevel 1 (
  echo [ERROR] Could not create backend\.env
  goto :fail
)

rem Pick SQLite when no PostgreSQL is listening on localhost:5432.
"%PY%" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',5432))==0 else 1)" 2>nul
if not errorlevel 1 goto :env_done

echo       No PostgreSQL on localhost:5432 - switching DATABASE_URL to SQLite.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%BACKEND%\.env';$t=[IO.File]::ReadAllText($p);$q=[char]34;$t=[regex]::Replace($t,'(?m)^DATABASE_URL=.*$','DATABASE_URL='+$q+'sqlite:///ckgai_dev.db'+$q);[IO.File]::WriteAllText($p,$t,(New-Object System.Text.UTF8Encoding($false)))"
if errorlevel 1 echo       [WARN] Could not patch backend\.env - set DATABASE_URL by hand.
:env_done

rem ---------------------------------------------------------- 5. frontend deps
if exist "%FRONTEND%\node_modules\next" (
  echo [5/6] Frontend dependencies already installed.
  goto :fe_done
)

echo [5/6] Installing frontend dependencies (npm install)...
pushd "%FRONTEND%"
call npm install
if errorlevel 1 (
  popd
  echo [ERROR] npm install failed.
  goto :fail
)
popd
echo [5/6] Frontend dependencies installed.
:fe_done

rem ------------------------------------------------------------- 6. seed data
if "%SKIP_SEED%"=="1" (
  echo [6/6] SKIP_SEED=1 - seeding skipped.
  goto :seed_done
)
if exist "%BACKEND%\.seeded" (
  echo [6/6] Database already seeded.
  goto :seed_done
)

echo [6/6] Seeding development data (first run)...
pushd "%BACKEND%"
"%PY%" -m scripts.seed
if errorlevel 1 (
  popd
  echo [WARN] Seeding did not complete - the database probably already has data.
  echo        Delete backend\.seeded to force another attempt.
  echo seeded-attempt> "%BACKEND%\.seeded"
  goto :seed_done
)
popd
echo seeded> "%BACKEND%\.seeded"
echo [6/6] Seed complete.
:seed_done

rem ------------------------------------------------------------- 7. launch
if "%DRY_RUN%"=="1" (
  echo.
  echo DRY_RUN=1 - skipping service windows.
  goto :summary
)

echo.
echo Launching service windows...
echo.

start "CKGAI - Backend (FastAPI :8000)" /d "%BACKEND%" cmd /k ""%PY%" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

start "CKGAI - ML plane (warmup)" /d "%BACKEND%" cmd /k ""%PY%" -c "import importlib.util as u;print('ML deps:',{m:bool(u.find_spec(m)) for m in ['torch','transformers','sentence_transformers','sklearn','scispacy','neo4j','google.genai']});from ml.warmup import warm;warm();print('ML warmup finished - models cached. Press Enter to close.');input()""

start "CKGAI - Frontend (Next.js :3000)" /d "%FRONTEND%" cmd /k npm run dev

rem Wait for the API before opening the browser.
echo Waiting for the backend health check...
set /a "tries=0"
:wait_health
set /a "tries+=1"
if %tries% gtr 60 goto :health_timeout
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health/live' -TimeoutSec 2 | Out-Null } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 goto :healthy
ping -n 3 127.0.0.1 >nul
goto :wait_health

:healthy
echo Backend is healthy - opening the frontend.
start "" "http://localhost:3000"
goto :summary

:health_timeout
echo [WARN] Backend did not answer within 120s.
echo        Check the "CKGAI - Backend" window for errors, then open
echo        http://localhost:3000 manually.

rem ------------------------------------------------------------- 8. summary
:summary
echo.
echo ==========================================================================
echo   RUNNING
echo     Frontend  http://localhost:3000
echo     Backend   http://localhost:8000        (docs  http://localhost:8000/docs)
echo     Health    http://localhost:8000/health/live
echo     ML plane  warmup window + ML_ENABLED inside the API
echo.
echo   OPTIONAL services (everything degrades gracefully without them):
echo     Redis     docker run -d -p 6379:6379 redis:7-alpine
echo     Postgres  docker compose -f backend\docker-compose.yml up postgres
echo     ML deps   set INSTALL_ML=1 and run start_all.bat again
echo.
echo   CLOSE: close each service window; this window can be closed now.
echo ==========================================================================
endlocal
exit /b 0

:fail
echo.
echo ==========================================================================
echo   START FAILED - see the message above.
echo ==========================================================================
pause
endlocal
exit /b 1
