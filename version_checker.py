import re
import pkg_resources
from collections import OrderedDict

def get_installed_versions():
    # Regular expression to extract package names from requirements.txt lines
    package_pattern = re.compile(r'^([a-zA-Z0-9_-]+)')

    installed_versions = OrderedDict()
    missing_packages = []

    try:
        # Read existing requirements_ver.txt if it exists
        try:
            with open('requirements_ver.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '==' in line:
                        package_name, version = line.split('==')
                        installed_versions[package_name] = version
        except FileNotFoundError:
            pass  # If file doesn't exist, proceed without loading

        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Extract package name from the line
                match = package_pattern.match(line)
                if not match:
                    continue  # Skip lines that don't match package pattern

                package_name = match.group(1)

                try:
                    # Get installed version
                    version = pkg_resources.get_distribution(package_name).version
                    installed_versions[package_name] = version
                except pkg_resources.DistributionNotFound:
                    if package_name not in installed_versions:
                        missing_packages.append(package_name)

    except FileNotFoundError:
        print("Error: requirements.txt file not found")
        return

    # Write results to requirements_ver.txt
    with open('requirements_ver.txt', 'w') as f:
        for package_name, version in installed_versions.items():
            f.write(f"{package_name}=={version}\n")

    # Print summary
    print(f"Successfully wrote {len(installed_versions)} packages to requirements_ver.txt")
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")

if __name__ == "__main__":
    get_installed_versions()
