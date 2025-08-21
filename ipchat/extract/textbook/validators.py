def assert_textbook_coverage(chapter):
    # minimal sanity checks
    assert chapter.chapter_metadata.title
    # every table/figure should have a page if present
    for t in chapter.structured_data.get("tables", []):
        assert t.page is not None
    for f in chapter.structured_data.get("figures", []):
        assert f.page is not None