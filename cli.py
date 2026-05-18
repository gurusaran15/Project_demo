#!/usr/bin/env python3
"""MoJ Dashboard CLI — skeleton to gradually replace the Makefile."""

import os
import subprocess
import sys

from utils import backend_run, ensure_backend_deps, ensure_frontend_deps, frontend_run, get_aws_env, get_commit

# ---------------------------------------------------------------------------
# TESTING & CODE QUALITY COMMANDS
# ---------------------------------------------------------------------------

def test_backend():
    """Run backend tests — mirrors `make test_backend`."""
    ensure_backend_deps()
    test = os.environ.get("TEST")
    backend_run(["poetry", "run", "pytest", test if test else "tests/"],
                env={**os.environ, "PYTHONPATH": "."})


def test_frontend():
    """Run frontend tests — mirrors `make test_frontend`."""
    ensure_frontend_deps()
    frontend_run(["npm", "run", "test"])


def fmt():
    """Run all formatters — mirrors `make fmt`."""
    print("Running prettier...")
    subprocess.run(["npx", "prettier", "--write", "."], cwd="frontend")
    print("Running TypeScript linting...")
    subprocess.run(["npm", "run", "lint"], cwd="frontend")
    print("Running TypeScript type checking...")
    subprocess.run(["npm", "run", "type-check"], cwd="frontend")
    ensure_backend_deps()
    print("Running isort...")
    subprocess.run(["poetry", "run", "isort", "--sp", "pyproject.toml", "."], cwd="backend")
    print("Running black...")
    subprocess.run(["poetry", "run", "black", "--config", "pyproject.toml", "."], cwd="backend")
    print("Running ruff...")
    backend_run(["poetry", "run", "ruff", "check", "--fix", "app", "tests"])


def enforce_types():
    """Enforce TypeScript types match backend Pydantic models — mirrors `make enforce_types`."""
    print("Enforcing type consistency between backend and frontend...")
    ensure_backend_deps()
    backend_run(["poetry", "run", "python", "../scripts/enforce_types.py"],
                env={**os.environ, "PYTHONPATH": "../backend"})


def test_env_setup():
    """Test complete environment setup — mirrors `make test_env_setup`."""
    print("Testing complete environment setup validation...")
    ensure_backend_deps()
    print("Running environment validation (GitHub, manual tasks, Docker)...")
    backend_run(["poetry", "run", "ansible-playbook", "../dev-tools/env-setup/env-setup-playbook.yml"])


# ---------------------------------------------------------------------------
# DATA SNAPSHOT MANAGEMENT COMMANDS
# ---------------------------------------------------------------------------

def snapshot_clean():
    """Clean all snapshot storage and build workspace — mirrors `make snapshot_clean`."""
    print("Cleaning snapshot storage and build workspace...")
    subprocess.run(["rm", "-rf", "dev_cache_storage/", "build_workspace", "build_workspace.zip"])
    print("Snapshot cleanup complete")


def snapshot_info():
    """Show snapshot storage information — mirrors `make snapshot_info`."""
    print("Snapshot storage information:")
    storage_type = os.environ.get("SNAPSHOT_STORAGE_TYPE", "local")
    print(f"Storage type: {storage_type}")
    if storage_type == "s3":
        print(f"S3 bucket: {os.environ.get('SNAPSHOT_S3_BUCKET', 'not set')}")
        print(f"S3 prefix: {os.environ.get('SNAPSHOT_S3_PREFIX', 'snapshot')}")
    else:
        print(f"Storage path: {os.environ.get('SNAPSHOT_STORAGE_PATH', './dev_cache_storage')}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    print(f"Current commit: {commit.stdout.strip()[:12]}")
    if storage_type == "local":
        storage_path = os.environ.get("SNAPSHOT_STORAGE_PATH", "./dev_cache_storage")
        if os.path.isdir(storage_path):
            result = subprocess.run(["ls", "-la", storage_path], capture_output=True, text=True)
            print(result.stdout or "  No snapshots found")
        else:
            print("  Storage directory does not exist")
    else:
        print("S3 snapshot info requires AWS CLI access (use aws s3 ls to check manually)")
    if os.path.isdir("./build_workspace"):
        print("  Build directory exists")
        if os.path.isfile("./build_workspace/meta.json"):
            result = subprocess.run(["stat", "-f", "%Sm", "./build_workspace/meta.json"], capture_output=True, text=True)
            print(f"  Last build: {result.stdout.strip()}")
    else:
        print("  Build directory does not exist")


def snapshot_build():
    """Build snapshot — mirrors `make snapshot_build`."""
    storage_type = os.environ.get("SNAPSHOT_STORAGE_TYPE", "local")
    env = get_aws_env()
    env["DEPLOYMENT_ID"] = get_commit()
    subprocess.run(["poetry", "install"], cwd="backend")
    if storage_type == "s3":
        s3_bucket = os.environ.get("SNAPSHOT_S3_BUCKET")
        if not s3_bucket:
            print("ERROR: SNAPSHOT_S3_BUCKET environment variable required for S3 storage")
            sys.exit(1)
        print(f"Building snapshot and uploading to S3 bucket: {s3_bucket}...")
        env.update({"SNAPSHOT_STORAGE_TYPE": "s3", "SNAPSHOT_S3_BUCKET": s3_bucket,
                    "SNAPSHOT_S3_PREFIX": os.environ.get("SNAPSHOT_S3_PREFIX", "snapshot")})
    else:
        print("Building snapshot locally using standalone source builders...")
        env.update({"SNAPSHOT_STORAGE_TYPE": "local", "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage"})
    backend_run(["poetry", "run", "python", "scripts/build_metrics_snapshot.py"], env=env)


# ---------------------------------------------------------------------------
# BACKEND & FRONTEND COMMANDS
# ---------------------------------------------------------------------------

def backend_run_cmd():
    """Start backend using existing snapshot — mirrors `make backend_run`."""
    storage_type = os.environ.get("SNAPSHOT_STORAGE_TYPE", "local")
    commit = get_commit()
    if storage_type == "s3":
        s3_bucket = os.environ.get("SNAPSHOT_S3_BUCKET")
        if not s3_bucket:
            print("ERROR: SNAPSHOT_S3_BUCKET environment variable required for S3 storage")
            sys.exit(1)
        env = {**os.environ, "SNAPSHOT_STORAGE_TYPE": "s3", "SNAPSHOT_S3_BUCKET": s3_bucket,
               "SNAPSHOT_S3_PREFIX": os.environ.get("SNAPSHOT_S3_PREFIX", "snapshot"), "DEPLOYMENT_ID": commit}
    else:
        env = {**os.environ, "SNAPSHOT_STORAGE_TYPE": "local",
               "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage", "DEPLOYMENT_ID": commit}
    backend_run(["poetry", "run", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], env=env)


def backend_build_run():
    """Build snapshot then start backend — mirrors `make backend_build_run`."""
    print("Building snapshot then starting backend...")
    snapshot_build()
    print("Snapshot built successfully, starting backend server...")
    backend_run_cmd()


def frontend():
    """Start frontend development server — mirrors `make frontend`."""
    print("Setting up frontend environment...")
    ensure_frontend_deps()
    print("Starting frontend development server...")
    frontend_run(["npm", "run", "dev"])


# ---------------------------------------------------------------------------
# DEPLOYMENT & DOCKER COMMANDS
# ---------------------------------------------------------------------------

def _docker_compose(args: list, env: dict) -> subprocess.CompletedProcess:
    """Run a docker-compose command in the dev-tools directory."""
    dev_tools = os.path.join(os.getcwd(), "dev-tools")
    return subprocess.run(["docker-compose"] + args, cwd=dev_tools, env=env)


def _local_docker_env() -> dict:
    """Build env dict for local Docker runs."""
    return {**os.environ, "DEPLOYMENT_ID": get_commit(),
            "SNAPSHOT_STORAGE_TYPE": "local", "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage"}


def docker_dev():
    """Start development environment — mirrors `make docker_dev`."""
    storage_type = os.environ.get("SNAPSHOT_STORAGE_TYPE", "local")
    print(f"Starting development environment with {storage_type} storage...")
    if storage_type == "s3":
        s3_bucket = os.environ.get("SNAPSHOT_S3_BUCKET")
        if not s3_bucket:
            print("ERROR: SNAPSHOT_S3_BUCKET environment variable required for S3 storage")
            sys.exit(1)
        env = get_aws_env()
        env.update({"DEPLOYMENT_ID": get_commit(), "SNAPSHOT_STORAGE_TYPE": "s3",
                    "SNAPSHOT_S3_BUCKET": s3_bucket,
                    "SNAPSHOT_S3_PREFIX": os.environ.get("SNAPSHOT_S3_PREFIX", "snapshot")})
    else:
        env = _local_docker_env()
    _docker_compose(["down", "--remove-orphans"], env)
    result = _docker_compose(["up"], env)
    sys.exit(result.returncode)


def docker_dev_full_stack():
    """Build fresh snapshot then start Docker — mirrors `make docker_dev_full_stack`."""
    print("Starting batteries included development environment...")
    print("Step 1: Building fresh snapshot from Athena...")
    snapshot_build()
    print("Step 2: Starting Docker development environment...")
    docker_dev()


def docker_dev_from_zip():
    """Start development environment from a zip file — mirrors `make docker_dev_from_zip`."""
    zipfile = os.environ.get("ZIPFILE")
    if not zipfile:
        print("ERROR: ZIPFILE parameter required. Usage: ZIPFILE=path/to/snapshot.zip python scripts/cli.py docker_dev_from_zip")
        sys.exit(1)
    if not os.path.isfile(zipfile):
        print(f"ERROR: Zip file {zipfile} does not exist")
        sys.exit(1)
    print(f"Using zip file: {zipfile}")
    snapshot_clean()
    os.makedirs("dev_cache_storage", exist_ok=True)
    subprocess.run(["cp", zipfile, "dev_cache_storage/downloaded_snapshot.zip"])
    print("Snapshot setup complete")
    env = {**_local_docker_env(), "SNAPSHOT_ALLOW_FALLBACK": "true"}
    _docker_compose(["down", "--remove-orphans"], env)
    result = _docker_compose(["up"], env)
    sys.exit(result.returncode)


def docker_clean():
    """Force clean all Docker containers and networks — mirrors `make docker_clean`."""
    print("Force cleaning all Docker containers and networks...")
    _docker_compose(["down", "--remove-orphans", "--volumes"], _local_docker_env())
    subprocess.run(["docker", "system", "prune", "-f", "--filter", "label=com.docker.compose.project=dev-tools"])
    print("Docker cleanup complete")


def deploy_fallback_emails():
    """Copy fallback emails file to pod — mirrors `make deploy_fallback_emails`."""
    pod = os.environ.get("POD")
    if not pod:
        print("ERROR: POD parameter required. Usage: POD=your-pod-name python scripts/cli.py deploy_fallback_emails")
        sys.exit(1)
    print(f"Copying fallback emails file to pod {pod}...")
    result = subprocess.run(["kubectl", "cp", "backend/config/fallback_emails.txt",
                             f"{pod}:/app/config/fallback_emails.txt"])
    print("File deployed successfully!")
    sys.exit(result.returncode)


def copy_fallback_emails():
    """Copy fallback emails file from pod to local — mirrors `make copy_fallback_emails`."""
    pod = os.environ.get("POD")
    if not pod:
        print("ERROR: POD parameter required. Usage: POD=your-pod-name python scripts/cli.py copy_fallback_emails")
        sys.exit(1)
    print(f"Copying fallback emails file from pod {pod}...")
    subprocess.run(["kubectl", "cp", f"{pod}:/app/config/fallback_emails.txt",
                    "backend/config/fallback_emails.txt"])
    print("Current contents:")
    result = subprocess.run(["cat", "backend/config/fallback_emails.txt"])
    sys.exit(result.returncode)


def k8s_latest_snapshot_logs():
    """View logs from latest cache build job — mirrors `make k8s_latest_snapshot_logs`."""
    print("Finding cache build job in current namespace...")
    job = subprocess.run(["kubectl", "get", "jobs", "--sort-by=.metadata.creationTimestamp", "-o", "name"],
                         capture_output=True, text=True)
    jobs = [j for j in job.stdout.splitlines() if "cache-build-" in j]
    if not jobs:
        print("No cache build jobs found in current namespace")
        subprocess.run(["kubectl", "get", "jobs"])
        sys.exit(1)
    job_name = jobs[-1].split("/")[-1]
    print(f"Found job: {job_name}")
    pod = subprocess.run(["kubectl", "get", "pods", f"--selector=job-name={job_name}",
                          "-o", "jsonpath={.items[0].metadata.name}"], capture_output=True, text=True)
    pod_name = pod.stdout.strip()
    if not pod_name:
        print(f"No pod found for job {job_name}")
        subprocess.run(["kubectl", "describe", "job", job_name])
        sys.exit(1)
    print(f"Found pod: {pod_name}")
    pod_status = subprocess.run(["kubectl", "get", "pod", pod_name, "-o", "jsonpath={.status.phase}"],
                                 capture_output=True, text=True).stdout.strip()
    print(f"Pod status: {pod_status}")
    if pod_status == "Running":
        print("Following live logs (Ctrl+C to exit):")
        result = subprocess.run(["kubectl", "logs", "-f", pod_name])
    else:
        print("Showing logs from completed job:")
        result = subprocess.run(["kubectl", "logs", pod_name])
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# AWS & BEDROCK COMMANDS
# ---------------------------------------------------------------------------

def bedrock_eu_models():
    """Show AWS Bedrock models in EU regions — mirrors `make bedrock_eu_models`."""
    result = subprocess.run(["./scripts/bedrock_eu_cline_models.sh"])
    sys.exit(result.returncode)


def bedrock_eu_models_all():
    """Show all AWS Bedrock models in EU regions — mirrors `make bedrock_eu_models_all`."""
    result = subprocess.run(["./scripts/bedrock_eu_cline_models.sh", "--all"])
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# SECURITY & MAINTENANCE COMMANDS
# ---------------------------------------------------------------------------

def login():
    """Login to AWS SSO — mirrors `make login`."""
    print("Logging into AWS SSO...")
    aws_profile = os.environ.get("AWS_PROFILE")
    if not aws_profile:
        print("ERROR: AWS_PROFILE environment variable is not set.")
        sys.exit(1)
    result = subprocess.run(["aws", "sso", "login", "--profile", aws_profile])
    sys.exit(result.returncode)


def security_bandit():
    """Run bandit security scanner — mirrors `make security_bandit`."""
    print("Running bandit security scanner (matches CI configuration)...")
    ensure_backend_deps()
    backend_run(["poetry", "run", "bandit", "-c", "pyproject.toml", "-r", ".", "--format=screen"])


# ---------------------------------------------------------------------------
# API REFERENCE COMMANDS
# ---------------------------------------------------------------------------

def create_api_reference():
    """Create API reference files — mirrors `make create_api_reference`."""
    print("Creating API reference files...")
    result = subprocess.run(["python", "scripts/create_api_reference.py"])
    sys.exit(result.returncode)


def check_api_reference():
    """Check API against reference files — mirrors `make check_api_reference`."""
    print("Checking API reference...")
    result = subprocess.run(["python", "scripts/check_api_reference.py"])
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# COMMANDS registry
# ---------------------------------------------------------------------------

COMMANDS = {
    "test_backend": test_backend,
    "test_frontend": test_frontend,
    "fmt": fmt,
    "enforce_types": enforce_types,
    "test_env_setup": test_env_setup,
    "snapshot_clean": snapshot_clean,
    "snapshot_info": snapshot_info,
    "snapshot_build": snapshot_build,
    "backend_run": backend_run_cmd,
    "backend_build_run": backend_build_run,
    "frontend": frontend,
    "docker_dev": docker_dev,
    "docker_dev_full_stack": docker_dev_full_stack,
    "docker_dev_from_zip": docker_dev_from_zip,
    "docker_clean": docker_clean,
    "deploy_fallback_emails": deploy_fallback_emails,
    "copy_fallback_emails": copy_fallback_emails,
    "k8s_latest_snapshot_logs": k8s_latest_snapshot_logs,
    "bedrock_eu_models": bedrock_eu_models,
    "bedrock_eu_models_all": bedrock_eu_models_all,
    "login": login,
    "security_bandit": security_bandit,
    "create_api_reference": create_api_reference,
    "check_api_reference": check_api_reference,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command not in COMMANDS:
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[command]()
