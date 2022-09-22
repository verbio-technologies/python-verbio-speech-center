# asr4

ASR based on Transformer DNNs, with multilingual and unsupervised information.

## Installation and test

Installation separed in three parts: testing, using the server, using the client.

### Client installation

This installation assumes you are working on python 3.9.

pip install -r requirements.client.txt

PYTHONPATH=/asr4/ python bin/client.py --host 192.168.2.4:50051 -a <path to your file>.wav

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