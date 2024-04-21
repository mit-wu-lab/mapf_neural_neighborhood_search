#include "common.h"

ostream &operator<<(ostream &out, const State &s) {
    return out << s.location << "," << s.timestep << "," << s.orientation;
}

ostream &operator<<(ostream &out, const Path &path) {
    for (const auto &s : path)
        out << "(" << s << ")->";
    return out << endl;
}

ostream &operator<<(ostream &out, const Constraint &constraint) {
    const auto [agent, loc1, loc2, t, positive] = constraint;
    return out << "<" << agent << "," << loc1 << "," << loc2 << "," << t << "," << positive << ">";
}

ostream &operator<<(ostream &out, const Conflict &conflict) {
    const auto [a1, a2, loc1, loc2, timestep] = conflict;
    return out << "<" << a1 << "," << a2 << "," << loc1 << "," << loc2 << "," << timestep << ">";
}

ostream &operator<<(ostream &out, const Interval &interval) {
    const auto [tmin, tmax, has_conflict] = interval;
    return out << "[" << tmin << "," << tmax << ")(" << has_conflict << ")";
}
