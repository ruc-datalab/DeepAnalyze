# SimpleCoder

This is a simple coding environment that allows solving SWE bench like coding challenges.

It uses a basic sandbox environment powered by Guix
(https://guix.gnu.org/), a package manager that comes with thousands
of pre-built software packages that you can easily customize and
compile.

## How it works

Guix makes it easy to set up development environments with all the right dependencies. For example, if you want to work on a Linux package like Inkscape, you can run `guix shell --development inkscape` and it automatically gives you a shell with everything needed to build that software.

You can also create custom environments by running `guix shell -m manifest.scm`, where the `manifest.scm` file lists exactly which packages and dependencies you want. Think of this like Python's `uv run` command, but instead of just handling Python projects, it can set up complete Linux development environments for any type of software.

## How to use it

The docker image / environment you are working in needs to have guix installed, you can
e.g. install it by running

```shell
wget https://guix.gnu.org/install.sh
chmod +x install.sh
sudo ./install.sh
```

For the following commands, we assume that you are in the `sky-train/examples/simplecoder` directory. It is worth running the following command once to initialize the packages (this ensures the following commands won't time out):

```shell
guix shell -m manifest.scm -- sh
```

To run the example, first clone the test repository
```shell
git clone https://github.com/SWE-agent/test-repo
```

and then run
```shell
python simplecoder.py
```

*Disclaimer*: The integration with SkyRL Train is still ongoing.

