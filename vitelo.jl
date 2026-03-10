#! /Users/servedio/.juliaup/bin/julia

using CSV, DataFrames, Statistics, Dates;

### Global Variables ###
"number of past days included in the statistics"
DAYRANGE = 999;

"minimum number of played games to enter the ranking"
MINNRGAMES = 10;

"If non-active in these last days do not appear in the stats"
INACTIVITY = 7;
########################

mutable struct Player
    name::String;
    rd::Vector{Int}; # score as red defender
    bd::Vector{Int}; # score as blue defender
    rf::Vector{Int}; # score as red forward
    bf::Vector{Int}; # score as blue forward
    wins::Vector{Int}; # won games
    playing_days::Set{Date}; # list of days the player played a match
    count::Int; # nr of games played
    avrg::Float64; # average score
    mul::Float64; # score modifier to apply to teammate
    atmwr::Float64; # Average Team Mate Win Rate
    entropy::Float64;
end

Player(name::AbstractString) = Player(name, Int[], Int[], Int[], Int[], Int[], Set{Date}(),0,0.0,1.0,0.0,0.0);

function entropy(df::DataFrame, P::Dict{String,Player})
    nplayers = length(unique([df.rd;df.rf;df.bd;df.bf]));
    for player in keys(P)
        MateFreq = Dict{String,Int}();
        dfil = filter(x->player ∈ [x.rd,x.rf,x.bd,x.bf], df);
        for r in eachrow(dfil)
            r.rd==player && (MateFreq[r.rf] = get(MateFreq, r.rf, 0) + 1);
            r.rf==player && (MateFreq[r.rd] = get(MateFreq, r.rd, 0) + 1);
            r.bd==player && (MateFreq[r.bf] = get(MateFreq, r.bf, 0) + 1);
            r.bf==player && (MateFreq[r.bd] = get(MateFreq, r.bd, 0) + 1);
        end
        tot = sum(values(MateFreq));
        ent = sum(x->-x/tot*log(x/tot), values(MateFreq))/log(nplayers-1);
        P[player].entropy = ent;
 
    end
end

function atmwr(df::DataFrame, P::Dict{String,Player})
    for player in keys(P)
        # find teammates
        TeamMates = String[];
        WinRates = Float64[];
        dfil = filter(x->player ∈ [x.rd,x.rf,x.bd,x.bf], df);
        for r in eachrow(dfil)
            r.rd==player && push!(TeamMates,r.rf); 
            r.rf==player && push!(TeamMates,r.rd); 
            r.bd==player && push!(TeamMates,r.bf); 
            r.bf==player && push!(TeamMates,r.bd); 
        end
        

        dfil = filter(x->player ∉ [x.rd,x.rf,x.bd,x.bf], df);
        
        for tm in unique(TeamMates) 
            wins = c = 0;
            matches = count(==(tm), TeamMates); # nr of games played with team mate
            dfil2 = filter(x->tm ∈ [x.rd,x.rf,x.bd,x.bf], dfil);
            for r in eachrow(dfil2)
                if tm ∈ [r.rd,r.rf] && r.resultr>r.resultb
                    wins += 1;
                end
                if tm ∈ [r.bd,r.bf] && r.resultr<r.resultb
                    wins += 1;
                end
                c += 1;
            end
            c>0 && push!(WinRates, matches*wins/c);
            # c>0 && player == "Simone" && println("Simone's TMs: $tm with WR=$(wins/c) and $matches games");
        end

        P[player].atmwr = sum(WinRates)/length(TeamMates);
    end
    return;
end

function maxMultiplier(df::DataFrame)
    s = 0;
    for r in eachrow(df)
        m = min(10,min(r.resultb, r.resultr));
        s += m;
    end
    avrgs = s / nrow(df);
    println("Average losing goals: $avrgs");
    return 10/avrgs;
end

function main(args::Vector{String})
    # if length(args) > 0
    #     file = args[1];
    # else
    #     file = "scores.csv";
    # end

    # Use the first argument passed from Python, or default to results.csv
    filename = length(ARGS) > 0 ? ARGS[1] : "results.csv"

    df = CSV.read(filename, DataFrame);
    dropmissing!(df);
    # display(df)
    # red defender := rd, blue forward := bf
    rename!(df, [:date, :rd, :rf, :bd, :bf, :resultr, :resultb]);
    # Line 112: Map the 7 columns in the order they come from Python
    transform!(df, :date => ByRow(x->Date(x, "dd-mm-yyyy")) => :date);
    # Use yyyy-mm-dd to match the ISO standard sent from Python
    # transform!(df, :date => ByRow(x -> Date(string(x), "yyyy-mm-dd")) => :date);
    maxmultiplier = maxMultiplier(df);
    filter!(x->(today()-x.date <= Day(DAYRANGE)), df);

    P = Dict{String,Player}(); # P[name]=Player
    for r in eachrow(df)
        
        get!(P, r.rd, Player(r.rd));
        get!(P, r.rf, Player(r.rf));
        get!(P, r.bd, Player(r.bd));
        get!(P, r.bf, Player(r.bf));

        push!(P[r.rd].wins, ifelse(r.resultr>r.resultb, 1, 0));
        push!(P[r.rf].wins, ifelse(r.resultr>r.resultb, 1, 0));
        push!(P[r.bd].wins, ifelse(r.resultr<r.resultb, 1, 0));
        push!(P[r.bf].wins, ifelse(r.resultr<r.resultb, 1, 0));

        push!(P[r.rd].playing_days, r.date);
        push!(P[r.rf].playing_days, r.date);
        push!(P[r.bd].playing_days, r.date);
        push!(P[r.bf].playing_days, r.date);
        
        resultr = min(r.resultr, 10); # ignore goals after 10
        resultb = min(r.resultb, 10);
      
        push!(P[r.rd].rd, resultr);
        push!(P[r.rf].rf, resultr);
        push!(P[r.bd].bd, resultb);
        push!(P[r.bf].bf, resultb);

    end

    # determine players' score average
    for (name,p) in P
        v = vcat(p.rd,p.rf,p.bd,p.bf);
        P[name].avrg = mean(v);
        P[name].count = length(v);
    end

    # select players with at least MINNRGAMES played games
    v = [(name, P[name].avrg, P[name].count) for name in keys(P)];
    vtop = filter(x->x[3]>=MINNRGAMES, v);
    
    # sort players with at least MINNRGAMES by average scores
    sort!(vtop, by=x->x[2], rev=true);

    topplayers = DataFrame(vtop);

    # logarithmically assign multipliers
    N = nrow(topplayers);
    topplayers.mul = maxmultiplier .^ (range(0,N,N) ./ N);

    for name in keys(P) #set default multiplier
        P[name].mul = sqrt(maxmultiplier);
    end
    for r in eachrow(topplayers)
        P[r[1]].mul = r.mul;
    end

    F = Dict{String,Vector{Float64}}(); # F[name]= Vector of weighted scores

    for r in eachrow(df)
        resultr = min(r.resultr, 10); # ignore goals after 10
        resultb = min(r.resultb, 10);

        get!(F, r.rd, Float64[]);
        get!(F, r.rf, Float64[]);
        get!(F, r.bd, Float64[]);
        get!(F, r.bf, Float64[]);
        # weight player's score with teammate's and direct opponent's multiplier 
        
        # push!(F[r.rd], resultr * P[r.rf].mul / P[r.bf].mul);
        # push!(F[r.rf], resultr * P[r.rd].mul / P[r.bd].mul);
        # push!(F[r.bd], resultb * P[r.bf].mul / P[r.rf].mul);
        # push!(F[r.bf], resultb * P[r.bd].mul / P[r.rd].mul);

        # push!(F[r.rd], resultr * P[r.rf].mul / P[r.bf].mul * P[r.rd].mul / P[r.bd].mul);
        # push!(F[r.rf], resultr * P[r.rd].mul / P[r.bd].mul * P[r.rf].mul / P[r.bf].mul);
        # push!(F[r.bd], resultb * P[r.bf].mul / P[r.rf].mul * P[r.bd].mul / P[r.rd].mul);
        # push!(F[r.bf], resultb * P[r.bd].mul / P[r.rd].mul * P[r.bf].mul / P[r.rf].mul);

        # push!(F[r.rd], resultr * sqrt(P[r.rf].mul / P[r.bf].mul * P[r.rd].mul / P[r.bd].mul));
        # push!(F[r.rf], resultr * sqrt(P[r.rd].mul / P[r.bd].mul * P[r.rf].mul / P[r.bf].mul));
        # push!(F[r.bd], resultb * sqrt(P[r.bf].mul / P[r.rf].mul * P[r.bd].mul / P[r.rd].mul));
        # push!(F[r.bf], resultb * sqrt(P[r.bd].mul / P[r.rd].mul * P[r.bf].mul / P[r.rf].mul));

        push!(F[r.rd], resultr * P[r.rf].mul / sqrt( P[r.bf].mul * P[r.bd].mul));
        push!(F[r.rf], resultr * P[r.rd].mul / sqrt( P[r.bd].mul * P[r.bf].mul));
        push!(F[r.bd], resultb * P[r.bf].mul / sqrt( P[r.rf].mul * P[r.rd].mul));
        push!(F[r.bf], resultb * P[r.bd].mul / sqrt( P[r.rd].mul * P[r.rf].mul));
    end

    vitelo = [round(Int,mean(x)*100) for x in values(F)];
    Δvitelo = [round(Int,std(x, corrected=false)/sqrt(length(x))*100) for x in values(F)];
    lv = [length(x) for x in values(F)];
    mul = [P[x].mul for x in keys(F)];
    avrg = [P[x].avrg for x in keys(F)];
    winrate = [mean(P[x].wins) for x in keys(F)];
    atmwr(df,P);
    avrg_teammate_wr = [P[x].atmwr for x in keys(F)];
    entropy(df,P);
    ent = [P[x].entropy for x in keys(F)];
    inactivity = [today()-maximum(P[x].playing_days) for x in keys(F)];

    dout = DataFrame(player = collect(keys(F)), 
                    WR = round.(winrate, digits=3),
                    ATMWR = round.(avrg_teammate_wr, digits=3),
                    VitELO=vitelo, 
                    ΔVitELO=Δvitelo,
                    Games=lv, 
                    Entropy = round.(ent, digits=3),
                    average_score=round.(avrg, digits=2),
                    multiplier=round.(mul, digits=2),
                    inactivity=inactivity);

#    sort!(dout, [:VitELO, :WR, :average_score], rev=true);
    sort!(dout, [:WR, :VitELO, :average_score], rev=true);
    println(dout);
    open("VitELO.txt", "w") do OUT
        println(OUT, dout[dout.Games .>= MINNRGAMES .&& dout.inactivity .<= Day(INACTIVITY), :]);
        println(OUT, "Timeframe: $DAYRANGE days\tMinimum number of games: $MINNRGAMES\tInactivity filter: $INACTIVITY days");
    end
    # CSV.write("VitELO.csv", dout[dout.nr_of_games .>= MINNRGAMES, :]);

    dv = dout[dout.Games .>= MINNRGAMES .&& dout.inactivity .<= Day(INACTIVITY), :];
    select!(dv, :player, [:VitELO, :ΔVitELO] => ByRow((x,y)->x-y) => :VmS);
    sort!(dv, :VmS, rev=true);
    # println(dv[1:4,:]);
    # println("Best four [VitELO-ΔVitELO] players\n");

    CSV.write("VitELO.csv", dout)

    return dout;
end


dout = main(ARGS);
