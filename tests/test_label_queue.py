from blinkwatch.web import prioritize_underrepresented


def test_prioritizes_unlabelled_crops_suggested_as_less_labelled_subjects():
    crops = [
        {"id": 1, "suggestion": {"label": "Mango", "score": 0.9}},
        {"id": 2, "suggestion": {"label": "Gatto sconosciuto", "score": 0.6}},
        {"id": 3, "suggestion": None},
        {"id": 4, "suggestion": {"label": "Buccio", "score": 0.8}},
    ]

    ordered = prioritize_underrepresented(crops, {"Mango": 166, "Gatto sconosciuto": 1, "Buccio": 119})

    assert [crop["id"] for crop in ordered] == [2, 4, 1, 3]


def test_review_filter_returns_only_unlabelled_crops_for_predicted_subject():
    import sqlite3
    from blinkwatch.web import review_candidates

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE crops (id, kind, label, pred_model, pred_label, auto_label, pred_margin, auto_margin)")
    db.executemany("INSERT INTO crops VALUES (?,?,?,?,?,?,?,?)", [
        (1, "persona", None, "big", "Io", "Mamma", 0.7, 0.6),
        (2, "persona", None, None, "Mamma", None, 0.2, None),
        (3, "persona", "Io", "big", "Io", None, 0.9, None),
        (4, "animale", None, "big", "Io", None, 0.1, None),
    ])

    rows = review_candidates(db, "persona", "Io", 10)

    assert [row["id"] for row in rows] == [1]
