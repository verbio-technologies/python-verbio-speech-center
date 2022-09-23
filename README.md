# asr4

ASR based on Transformer DNNs, with multilingual and unsupervised information.

## Installation and test

Installation separed in two parts: installing the server, and installing the client. The client having many less dependencies than the server.

### Client installation

The client requires python version at least 3.9. To install the requirementes do this from the root of the `asr4` repo:

```sh
pip install -r requirements.client.txt
```

Once installed, the client can connect to a running `asr4` server to obtain transcriptions. This simple command will return the transcription through the standard output channel:

```sh
PYTHONPATH=<path to asr4> python bin/client.py --host 192.168.2.4:50051 -a <path to your file>.wav
```

Note that it needs to define `PYTHONPATH` to the root of the repo to work.



### Server installation

The server requires python version at least 3.9. To install the requirementes do this from the root of the `asr4` repo:

```sh
pip install pytest==6.2.5 torch==1.12.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export MAKEFLAGS="-j4"
python setup.py egg_info
sed -i 's/@ /@/g' *.egg-info/requires.txt
pip install `grep -v '^\[' *.egg-info/requires.txt`
pip install .
```


## Formatting & Linting

As a general note, you should always apply proper formatting and linting before pushing a commit. To do so, please run the following:

```sh
# Rust
cargo fmt --all # Formatting
cargo clippy --all --all-targets -- -D warnings # Linting

# Python
black . # Formatting & Linting
```

## How to generate a new release

In order to manage releases we will use `cargo-release`: https://github.com/sunng87/cargo-release

If you want to install it, run the following command:

```
$ cargo install cargo-release
```

:warning: If you want to see what operations it would apply, you can add the `--dry-run` flag and it won't actually perform any operation.

In order to generate a new release, you can invoke the following command (keep in mind the flag `--workspace` is required in order to update all the subcrates):

```
# Increment major
$ cargo release major --workspace

# Increment minor
$ cargo release minor --workspace

# Increment patch
$ cargo release patch --workspace

# Set specific version
$ cargo release 1.2.3 --workspace 
```

It will:

1. Set the new version to all the `Cargo.toml` of the main crate and subcrates. It also updates the `Cargo.lock`.
2. Set the new version to the `CMakeLists.txt` file.
3. Create a "Bump version" commit.

⚠ IT DOES NOT CREATE THE TAG NOR PUSH ANYTHING

So, a typical workflow would be the following:

```
$ cargo release minor --workspace
$ git push 
$ git tag VERSION_NUMBER
$ git push --tags 
```
The user must be able to perform both the `git push` to upload the commit and the `git push --tags` to upload the newly created tag.

Once the tag has been pushed, you should edit the tag's Release notes via the GitLab UI and add the changelog for the version.

Then, the CI system will start doing its job and it will generate a `.deb` package, and via webhook the package will be available at the Verbio staging repositories.
