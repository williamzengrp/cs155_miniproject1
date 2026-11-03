"""r2_eda: label-free features suggested by the error analysis (all computed on the train+test pool, no labels).

mdup block (date-mates' duplicate structure = "album has a multi-playlist song"):
  mate_multi      number of OTHER songs (distinct uri) on the same release date that have >= 2 copies in the pool
  mate_extra      number of extra copies among the other songs on that date  (rows - distinct songs, own song excluded)
  mate_maxcnt     max pool copy count among the other songs on that date (0 if no date-mates)
  mate_multi_frac mate_multi / number of other songs on the date
ucnt block: uri_cnt = pool copy count of the row's own song (uri used only to identify duplicates).
"""
import numpy as np
import pandas as pd


def mdup_block(tr, te):
    al = pd.concat([tr, te], ignore_index=True)
    raw = al['track_album_release_date'].astype(str)
    uri = al['uri']
    ucnt = uri.map(uri.value_counts())
    first = ~uri.duplicated()
    rows_on_date = raw.map(raw.value_counts())
    songs_on_date = raw.map(raw[first].value_counts())
    multi_on_date = raw.map(raw[first & (ucnt >= 2)].value_counts()).fillna(0)
    F = pd.DataFrame(index=al.index)
    F['mate_multi'] = (multi_on_date - (ucnt >= 2).astype(int)).astype(float)
    F['mate_extra'] = ((rows_on_date - ucnt) - (songs_on_date - 1)).astype(float)
    # max copy count among other songs on the date
    S = pd.DataFrame({'raw': raw[first], 'u': uri[first], 'c': ucnt[first]})
    top2 = S.sort_values('c', ascending=False).groupby('raw')['c'].apply(lambda s: list(s.iloc[:2]) + [0, 0])
    t1 = raw.map(top2.str[0])
    t2 = raw.map(top2.str[1])
    # if own song holds the max, take the second largest
    S2 = S.sort_values('c', ascending=False).groupby('raw')['u'].first()
    own_is_top = raw.map(S2) == uri
    F['mate_maxcnt'] = np.where(own_is_top, t2, t1).astype(float)
    F['mate_multi_frac'] = F['mate_multi'] / np.maximum(songs_on_date - 1, 1)
    return F


def ucnt_block(tr, te):
    al = pd.concat([tr, te], ignore_index=True)
    return pd.DataFrame({'uri_cnt': al['uri'].map(al['uri'].value_counts()).astype(float)})


BLOCKS = {'mdup': mdup_block, 'ucnt': ucnt_block}


def extra_block(tr, te, names):
    """Concatenate the requested blocks; returns (train part, test part)."""
    n = len(tr)
    if not names:
        return None, None
    P = pd.concat([BLOCKS[b](tr, te) for b in names], axis=1)
    return P.iloc[:n].reset_index(drop=True), P.iloc[n:].reset_index(drop=True)


def dround_block(tr, te):
    """Whole-second durations (fresh-eyes EDA): duration_ms that is an exact multiple of 1000 ms marks tracks from
    specific production / distribution pipelines (233 train rows, 22% High vs 36% otherwise, mostly 2020+).
      dur_tz          trailing zeros of round(duration_ms), capped at 5
      dur_round1000   duration_ms % 1000 == 0
      date_round_frac fraction of the OTHER distinct songs on the same release date with whole-second durations
    Label-free, computed on the train+test pool (uri only to de-duplicate songs)."""
    al = pd.concat([tr, te], ignore_index=True)
    d = al['duration_ms'].round().astype('int64')
    tz = np.zeros(len(d), int)
    v = d.to_numpy().copy()
    for _ in range(5):
        hit = (v % 10 == 0) & (v > 0)
        tz += hit
        v = np.where(hit, v // 10, 1)
    r1000 = (d % 1000 == 0).astype(int)
    raw = al['track_album_release_date'].astype(str)
    first = ~al['uri'].duplicated()
    songs = raw.map(raw[first].value_counts())
    rnd = raw.map(raw[first & (r1000 == 1)].value_counts()).fillna(0)
    F = pd.DataFrame(index=al.index)
    F['dur_tz'] = tz.astype(float)
    F['dur_round1000'] = r1000.astype(float)
    F['date_round_frac'] = ((rnd - r1000) / np.maximum(songs - 1, 1)).astype(float)
    return F


BLOCKS['dround'] = dround_block
