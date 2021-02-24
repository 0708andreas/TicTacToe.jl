using StatsBase

export Field, empty, cross, naught, TTTEnv, reset!, state, is_win, is_draw,
    is_done, TTTLearner, find_boards, train!, play_game, pp, eval_learner,
    play_against, play_and_learn

# First, let's get some data structures set up. A board is a $3 \times 3$ matrix with each field being either empty, or occupied by a cross or a naught. Let's define a Field as being either empty, having a cross or having a naught.
@enum Field empty cross naught
Board = Array{Field, 2}

# Let's give ourselves a utility-function to switch players.
function other_player(p::Field)
    @assert p != empty
    if p == cross
        return naught
    else
        return cross
    end
end

# In the above terminology the state-space is the space of all legal tic tac toe board configurations. The following function discovers every such board given a base-case of a board filled with `empty` and the starting player.
function find_boards(board::Board, player::Field)
    boards = Vector{}()
    append!(boards, [board])
    for i in 1:3
        for j in 1:3
            if board[i, j] == empty
                b = copy(board)
                b[i, j] = player
                push!(boards, b)
                for bb in find_boards(b, other_player(player))
                    if ! (bb in boards)
                        push!(boards, bb)
                    end
                end
            end
        end
    end
    return boards
end

# The action space is all coordinates $\{(i, j) | 1 \leq i, j \leq 3\}$. Julia has no way to enforce the size of the coordinates, so an action is simply a tuple (i, j) in Julia called a cartesian 2-index, and we'll take care to only construct valid coordinates in the rest of the code.
Action = CartesianIndex{2}

# Now, let's make a type for the environment. Notice, that this doesn't strictly follow the definition of an environment. I hope you will see later, that this style is actually very natural. We hold the important data in a tuple and define functions elsewhere to extract permissible actions, rewards termination, as well as an update function. This allows us to keep the interface decoupled from the representation of our data. Thus, if we need to change for instance the representation of a board, say for efficiency reasons, we can do so and only have to update a few functions instead of every place we use the data.
mutable struct TTTEnv
    state::Board
    reward::Float64
    done::Bool
end

# Now, let's define our functions. Notice that our functions now take an environment as first argument instead of a state. This has no significant consequences, it just makes for better organized code.
state(env::TTTEnv) = env.state
action_mask(env::TTTEnv) = vec([CartesianIndex(i, j) for i in 1:3 , j in 1:3])
reward(env::TTTEnv) = env.reward
terminated(env::TTTEnv) = env.done

# We'll wait a bit to define the update function, as we need some auxillary functions first. First, we need to figure out if we've won, lost or is in a draw.
function is_win(state::Board, player::Field)
    b = state .== player
    @inbounds begin
        b[1, 1] & b[1, 2] & b[1, 3] ||
            b[2, 1] & b[2, 2] & b[2, 3] ||
            b[3, 1] & b[3, 2] & b[3, 3] ||
            b[1, 1] & b[2, 1] & b[3, 1] ||
            b[1, 2] & b[2, 2] & b[3, 2] ||
            b[1, 3] & b[2, 3] & b[3, 3] ||
            b[1, 1] & b[2, 2] & b[3, 3] ||
            b[1, 3] & b[2, 2] & b[3, 1]
    end
end
function is_draw(state::Board)
    is_full = reduce(&, .! (state .== empty))
    has_winner = is_win(state, naught) || is_win(state, cross)
    return is_full && (! has_winner)
end
function is_done(state::Board)
    return is_draw(state) || is_win(state, cross) || is_win(state, naught)
end

# Now, we're ready to define $\epsilon$. We won't call it $\epsilon$, as it is morally an evaluation-function, so we'll say $env(action) = \epsilon(state(env), action)$.
function (env::TTTEnv)(a::Action)
# First we make our move
    if state(env)[a] != empty
        println(state(env), a)
    end
    
    @assert state(env)[a] == empty
    state(env)[a] = cross
# Then we check if the game is done.

    if is_win(state(env), cross)
        env.done = true
        env.reward = win_rew
        return
    elseif is_draw(state(env))
        env.done = true
        env.reward = draw_rew
        return
    end
# Now, let naught make a random move.
    naught_move = rand(findall(state(env) .== empty))
    env.state[naught_move] = naught
# Check if the game is done.
    if is_win(state(env), naught)
        env.done = true
        env.reward = loss_rew
    elseif is_draw(state(env))
        env.done = true
        env.reward = draw_rew
    end
end
# Notice that we havent' specified the rewards for winning, losing or getting a draw. This is because these are so-called hyper-parameters that we need to tweak in order for the learning to work properly. We'll discuss this later. For now, the important part is that a reward > 1 will reinforce the behaviour that lead to this reward and a reward < 1 will punish it. Be aware that this is not intrinsic to reinforcement learning, it just makes our naive training process easier. You'll see.

win_rew = 1.2
draw_rew = 0.2
loss_rew = 0.1

# That's our update function. This is usually where the domain-knowledge about the problem lives. If we navigate a physical space, this contains the physics simulation. In this case we simulate an opponent.
#
# Before we move on, let's just define a simple function to reset an evnironment before a new game.
function reset!(env::TTTEnv)
    env.state = fill(empty, (3, 3))
    env.reward = 0
    env.done = false
end


# That was the environment. It wasn't too bad, was it? Now, we're getting into the agent or learner and this is where the magic happens. First, I have to decide how to model what I've learned. For a very simple first model, let's just keep track of every possible board and try to learn what the best move is in each. Since tic tac toe isn't that big of a game, this is a feasible strategy. To facilitate learning, we'll keep track of how likely we think it is that each field is the best one to pick. This would look something like this:

#=
# +-----------+           +--------------------+
# | x |   | o |           | 0.00 | 0.03 | 0.00 |
# | x |   | o |   |---->  | 0.00 | 0.05 | 0.00 |
# |   |   |   |           | 0.50 | 0.02 | 0.40 |
# +-----------+           +--------------------+
=#

# This picture would tell us that bottom left corner is probably the best choice, but bottom right is probably also good.
#
# The idea is to try some games, figure out what works and what doesn't and then update the probabilities somehow. With that in mind, let's define an agent to simply be such a map from states to probabilities:
mutable struct TTTLearner
    model::Dict{Board, Array{Float64, 2}}
end


# Now, we need to populate this map with some initial values. To do this, I've made this little function that uses the statespace-function from before and initializes everything with random probabilities.
function TTTLearner()
    boards = find_boards(fill(empty, (3, 3)), cross)
    model = Dict{Board, Array{Float64, 2}}()
    for b in boards
        is = findall(b .== empty)
        # We just write a random number to every empty position and then normalize such that the sum of the probabilities is one.
        weights = fill(1, length(is))
        weights = Base.:/.(weights, sum(weights))
        value = fill(0.0, (3, 3))
        value[is] = weights
        model[b] = value
    end
    return TTTLearner(model)
end

# Let's give life to our agent by giving it a will of it's own. To take an action, we simply sample from the positions with weights given by the learned probabilities. Notice that we naively sample from all positions, but that the probability of choosing non-empty fields will always be zero.
(learner::TTTLearner)(env::TTTEnv) = learner(state(env))
function (learner::TTTLearner)(state::Board)
    values = vec([CartesianIndex(i, j) for i in 1:3 , j in 1:3])
    probs = vec(learner.model[state])
    w = Weights(probs)
    return sample(values, w)
end


# Finally, the magic part. This is where the learning takes place after we playes through a game. There's a bit going on here, so let's work through it. The parameters for the function are:
# - A learner to train
# - A trajectory. This is lingo. It's just a record of every state we went through in a game
# - Actions holds which actions we took at each board in the trajectory
# - A reward, telling us if the game was good or bad
#
# The idea is the following: if we won, every step we took must have been at least somewhat good. Thus, we can multiply the probability that the action we took was good by the reward and renormalize the probabilities. Similarly is we loose, we multiply every action we took by the reward to learn that those were bad moves. This is where it's important to choose our rewards to be grater than 1 if we won and less than 1 if we lost.
function update!(learner::TTTLearner,
                 trajectory::Vector{Board},
                 actions::Vector{Action},
                 reward::Float64)
    @assert length(trajectory) == length(actions)
    for i in 1:(length(trajectory) - 1)
        b = trajectory[i]
        a = actions[i]
        learner.model[b][a] *= reward

        learner.model[b] /= sum(learner.model[b])
    end
end

# Let's talk a bit about the choice of rewards. As you can see, any reward grater than 1 will encourage the behaviour that lead to that reward and a reward less than 1 will discourage it. 

# That's the setup. All there's left to do is to play some games:
function play_game(env::TTTEnv, learner::TTTLearner)
    trajectory = Vector{Board}()
    actions = Vector{Action}()
    reset!(env)
    while ! env.done
        action = learner(env)
        push!(trajectory, copy(state(env)))
        push!(actions, action)
        env(action)
    end
    return (trajectory, actions)
end
# Most of this function is book-keeping to build the trajectory. Actually, a single round of a game can be expressed as simply `env(learner(env))`.

# To make things easier for ourselves, we'll create a function to play lots of games in a row and learn from them. By default, we play 1 000 000 games at a time.
function train!(learner::TTTLearner, env::TTTEnv ; n=1_000_000)
    for _ in 1:n
        trajectory, actions = play_game(env, learner)
        update!(learner, trajectory, actions, reward(env))
    end
end

# We want to see if our model has learned something. This function plays 1000 games and records the number of losses, draw and wins.
function eval_learner(learner::TTTLearner, env::TTTEnv ; n=1000)
    rewards = Vector{Float64}()
    for _ in 1:n
        play_game(env, learner)
        push!(rewards, reward(env))
    end
    losses = count(rewards .== -2)
    draws = count(rewards .== 1)
    wins = count(rewards .== 2)
    return Base.:/(sum(rewards), n), losses, draws, wins
end

# That's it, that's all it takes to teach a computer how to play tic tac toe. Just for fun, let's write a function to play againt it:

function play_against(learner::TTTLearner)
    board = fill(empty, (3, 3))
    trajectory = Vector{Board}()
    while ! is_done(board)
        board[learner(board)] = cross
        pp(board)
        if is_done(board)
            break
        end
        println("What's your move? ")
        s = readline()
        coords = split(s, ' ', keepempty = false)
        i = parse(Int, coords[1])
        j = parse(Int, coords[2])
        board[i, j] = naught
        push!(trajectory, copy(board))
    end
    return trajectory
end

function play_and_learn(learner::TTTLearner)
    trajectory = Vector{Board}()
    actions = Vector{Action}()
    board = fill(empty, (3, 3))
    while ! is_done(board)
        action = learner(board)
        push!(trajectory, copy(board))
        push!(actions, action)
        board[action] = cross
        pp(board)
        if is_done(board)
            break
        end
        println("What's your move? ")
        s = readline()
        coords = split(s, ' ', keepempty = false)
        i = parse(Int, coords[1])
        j = parse(Int, coords[2])
        board[i, j] = naught
    end
    reward = draw_rew
    if is_win(board, cross)
        reward = win_rew
    elseif is_win(board, naught)
        reward = loss_rew
    end

    update!(learner, trajectory, actions, reward)
end

# Aaand that's it. After just 1000 games our agent have learned to never lose. If you try to play against it you will notice some curious behaviour from the agent. Try letting it have some wins. Notice, that it often won't go directly for the win and instead secure more ways to win. This is not a problem, since it's going to win either way, and we never told it to win quickly. This teaches us a valuable lesson: we often don't know what behaviour we're encuraging until we see the result of the learning.


# That was fun. But a little unsatisfactory. This solution doesn't scale very well to larger problems.


function pp(board::Board)
    function to_text(f::Field)
        if f == empty
            "."
        elseif f == cross
            "x"
        else
            "o"
        end
    end
    display(map(to_text, board))
end
