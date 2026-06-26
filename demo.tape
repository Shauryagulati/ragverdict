# demo.tape — reproducible README gif for ragverdict.
#
# Regenerate:  vhs demo.tape   ->   writes docs/demo.gif
# Install VHS: brew install vhs   (https://github.com/charmbracelet/vhs)
#
# Run from the repo root, in a shell where `ragverdict` is importable.
# The Hide/Show block below activates the conda env invisibly — change
# `dev` to your env name, or delete the three lines if `ragverdict` is
# already on your PATH.

Output docs/demo.gif

Set Shell "zsh"
Set FontSize 18
Set Width 1240
Set Height 820
Set Padding 28
Set Theme "Dracula"
Set TypingSpeed 55ms

# --- setup (not recorded) ---
Hide
Type "conda activate dev" Enter
Sleep 800ms
Type "clear" Enter
Sleep 300ms
Show

# --- the shot ---
Sleep 600ms
Type "ragverdict run examples/demo_rag/config.yaml --no-judge"
Sleep 700ms
Enter
Sleep 5s
