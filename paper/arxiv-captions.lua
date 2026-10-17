-- VecForge — Universal Local-First Vector Database
-- Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
-- Licensed under the Business Source License 1.1 (BSL 1.1)

-- Pandoc Lua filter for the arXiv LaTeX build (pandoc 3.1.3 AST).
--
-- 1. Every figure and table caption in vecforge-paper.md starts with a bold
--    "Figure N." or "Table N." run, because the HTML and Word builds do not
--    number captions. LaTeX numbers them itself, so the prefix is removed here
--    and the number appears once.
-- 2. Image paths are reduced to the bare file name (figures/fig-x.png becomes
--    fig-x.png), because arXiv wants every figure at the root of the package.

local function is_number_prefix(inline)
  if inline.t ~= "Strong" then
    return false
  end
  local text = pandoc.utils.stringify(inline)
  return text:match("^Figure %d+%.$") ~= nil or text:match("^Table %d+%.$") ~= nil
end

-- Drop a leading "Figure N." or "Table N." run and the space after it.
local function strip_prefix(inlines)
  if #inlines == 0 or not is_number_prefix(inlines[1]) then
    return inlines
  end
  local out = pandoc.List()
  local skipping = true
  for i = 2, #inlines do
    local inline = inlines[i]
    if skipping and (inline.t == "Space" or inline.t == "SoftBreak") then
      -- still inside the gap after the prefix
    else
      skipping = false
      out:insert(inline)
    end
  end
  return out
end

local function strip_caption(caption)
  if caption == nil or caption.long == nil then
    return
  end
  for _, block in ipairs(caption.long) do
    if block.t == "Plain" or block.t == "Para" then
      block.content = strip_prefix(block.content)
    end
  end
end

function Figure(fig)
  strip_caption(fig.caption)
  return fig
end

function Table(tbl)
  strip_caption(tbl.caption)
  return tbl
end

function Image(img)
  img.src = img.src:match("([^/]+)$") or img.src
  img.caption = strip_prefix(img.caption)
  return img
end
