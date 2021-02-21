using TicTacToe
using Documenter

DocMeta.setdocmeta!(TicTacToe, :DocTestSetup, :(using TicTacToe); recursive=true)

makedocs(;
    modules=[TicTacToe],
    authors="andreas <andreas@lha66.dk> and contributors",
    repo="https://github.com/0708andreas/TicTacToe.jl/blob/{commit}{path}#{line}",
    sitename="TicTacToe.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://0708andreas.github.io/TicTacToe.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/0708andreas/TicTacToe.jl",
)
