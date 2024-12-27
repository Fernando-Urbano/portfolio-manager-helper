import requests
import re
import subprocess

# Configuration
PACKAGE_NAME = "portfolio-management"
VERSION_FILE = "version/current_version.txt"


# create AuthorizationError class
class AuthorizationError(Exception):
    pass

def get_version_from_pypi(package_name):
    """Get the current version of the package from PyPI."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data['info']['version']

def update_current_version_file(package_name, file_path):
    """Check the current version on PyPI and overwrite current_version.txt."""
    try:
        current_version = get_version_from_pypi(package_name)
        with open(file_path, 'w') as file:
            file.write(current_version)
        print(f"Current version {current_version} written to {file_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def increment_version(current_version, part):
    """Increment the version based on the specified part (major, minor, patch)."""
    major, minor, patch = map(int, current_version.split("."))
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError("Invalid part. Choose from: major, minor, patch.")
    return f"{major}.{minor}.{patch}"

def update_local_version_file(new_version):
    """Update the local version.py file with the new version."""
    with open(VERSION_FILE, "r") as f:
        lines = f.readlines()
    with open(VERSION_FILE, "w") as f:
        for line in lines:
            if "__version__" in line:
                f.write(f'__version__ = "{new_version}"\n')
            else:
                f.write(line)

def create_and_push_git_branch(new_version):
    """Create and push a new Git branch to GitHub."""
    branch_name = f"release/{new_version}"

    subprocess.run(["git", "checkout", "main"], check=True)
    subprocess.run(["git", "pull", "origin", "main"], check=True)

    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)

    print(f"New branch {branch_name} created and pushed to GitHub.")

if __name__ == "__main__":
    try:
        check = input("Please review with Fernando before proceeding. Press 'checked' to continue: ")
        if check.lower() != "checked":
            raise AuthorizationError("You are not authorized to proceed.")

        update_current_version_file('your-package-name', 'current_version.txt')
        pypi_version = get_version_from_pypi(PACKAGE_NAME)
        print(f"Current version on PyPI: {pypi_version}")

        increment_type = input("Increment which part? (major/minor/patch): ").strip().lower()
        new_version = increment_version(pypi_version, increment_type)
        print(f"New version: {new_version}")

        update_local_version_file(new_version)
        print(f"Updated local version to {new_version}")

        create_and_push_git_branch(new_version)
    except AuthorizationError as e:
        print(f"AuthorizationError: {e}")
    except Exception as e:
        print(f"Error: {e}")
