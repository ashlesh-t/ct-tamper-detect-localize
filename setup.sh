#!/bin/bash

# -------------------------------
# CONFIGURATION
# -------------------------------
PYTHON_VERSION="3.10.12"
PROJECT_NAME="$(basename "$PWD")"

# -------------------------------
# Function: install pyenv
# -------------------------------
install_pyenv() {
    echo "Installing pyenv..."
    curl https://pyenv.run | bash

    echo "Adding pyenv to shell config..."
    if [[ -n "$ZSH_VERSION" ]]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi

    grep -q 'pyenv init' "$SHELL_RC" || {
        echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> "$SHELL_RC"
        echo 'eval "$(pyenv init --path)"' >> "$SHELL_RC"
        echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"
        echo 'eval "$(pyenv virtualenv-init -)"' >> "$SHELL_RC"
    }

    echo "pyenv installed. Please restart your shell or run: exec \$SHELL"
}

# -------------------------------
# Function: install direnv
# -------------------------------
install_direnv() {
    echo "Installing direnv..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt install direnv -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install direnv
    else
        echo "Unsupported OS for auto-install. Install direnv manually."
        return
    fi

    echo "🛠️ Adding direnv hook to shell..."
    if [[ -n "$ZSH_VERSION" ]]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi

    grep -q 'direnv hook' "$SHELL_RC" || {
        echo 'eval "$(direnv hook bash)"' >> "$SHELL_RC"
    }

    echo "direnv installed. Please restart your shell or run: exec \$SHELL"
}

# -------------------------------
# Function: setup project envrc
# -------------------------------
setup_envrc() {
    echo "Setting up .envrc to use pyenv Python $PYTHON_VERSION..."

    echo "use python $PYTHON_VERSION" > .envrc

    echo "Cleaning old .python-version if exists..."
    rm -f .python-version

    echo "Allowing direnv..."
    direnv allow
}

# -------------------------------
# MAIN SCRIPT START
# -------------------------------

# Install pyenv if missing
if ! command -v pyenv >/dev/null 2>&1; then
    install_pyenv
else
    echo "pyenv already installed."
fi

# Install direnv if missing
if ! command -v direnv >/dev/null 2>&1; then
    install_direnv
else
    echo "direnv already installed."
fi

# Install Python version (if not already)
if ! pyenv versions --bare | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION via pyenv..."
    pyenv install "$PYTHON_VERSION"
else
    echo "Python $PYTHON_VERSION already installed via pyenv."
fi

# Set local version
echo "$PYTHON_VERSION" > .python-version
pyenv local "$PYTHON_VERSION"

# Setup .envrc
setup_envrc

# Create default files if missing
touch requirements.txt README.md .gitignore

echo "All done! You can now start developing in Python $PYTHON_VERSION"
echo "Run: python --version   (should match)"
