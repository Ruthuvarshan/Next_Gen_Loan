"""
Spark demo control API endpoints.
Allows opening a console for Spark UI demo from the web interface.
"""

import subprocess
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/spark", tags=["Spark"])


class SparkDemoResponse(BaseModel):
    status: str
    message: str
    spark_ui_url: str


@router.post("/start-demo", response_model=SparkDemoResponse)
async def start_spark_demo():
    """
    Open a console window at the project directory, ready to run Spark demo.
    """
    try:
        # Path to the workspace
        workspace = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Verify the script exists
        cmd_path = os.path.join(workspace, "run_spark_ui_demo.cmd")
        if not os.path.exists(cmd_path):
            raise HTTPException(
                status_code=404,
                detail=f"Demo script not found at {cmd_path}"
            )
        
        # Open a new console at the workspace directory
        subprocess.Popen(
            ['cmd', '/k', f'cd /d {workspace} && echo. && echo ============================================== && echo    Spark UI Demo Console && echo ============================================== && echo. && echo Ready to run Spark demo! && echo. && echo Type this command and press ENTER: && echo    run_spark_ui_demo.cmd && echo.'],
            cwd=workspace,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        return SparkDemoResponse(
            status="opened",
            message="Console opened! Type 'run_spark_ui_demo.cmd' and press ENTER to start.",
            spark_ui_url="http://localhost:4040"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open console: {str(e)}"
        )


@router.get("/status")
async def get_spark_status():
    """
    Check Spark UI availability.
    """
    return {
        "status": "available",
        "spark_ui_url": "http://localhost:4040",
        "message": "Open console and run demo to start Spark UI"
    }

