-- The docx writer drops the keywords field, so put a Keywords line after the abstract.
function Pandoc(doc)
  local kw = doc.meta.keywords
  if kw then
    local text = pandoc.utils.stringify(kw)
    table.insert(doc.blocks, 1, pandoc.Para({ pandoc.Strong({ pandoc.Str("Keywords:") }), pandoc.Space(), pandoc.Str(text) }))
  end
  return doc
end
