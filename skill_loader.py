import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SKILLS_DIR = PROJECT_DIR / "skills"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    directory: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md 必须以 YAML frontmatter 开始")

    try:
        end_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("SKILL.md 缺少 frontmatter 结束标记") from exc

    metadata = {}
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            raise ValueError(f"无法解析 frontmatter 行：{line}")
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    return metadata, "\n".join(lines[end_index + 1 :]).strip()


def load_skill(name: str, skills_dir: Path = DEFAULT_SKILLS_DIR) -> SkillDefinition:
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"非法 Skill 名称：{name!r}")

    directory = (skills_dir / name).resolve()
    root = skills_dir.resolve()
    if root not in directory.parents:
        raise ValueError("Skill 路径超出 skills 目录")

    skill_file = directory / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"找不到 Skill：{skill_file}")

    metadata, instructions = _parse_frontmatter(
        skill_file.read_text(encoding="utf-8")
    )
    skill_name = metadata.get("name", "")
    description = metadata.get("description", "")
    if skill_name != name or directory.name != name:
        raise ValueError("Skill name、目录名和调用名必须一致")
    if not description:
        raise ValueError("Skill description 不能为空")
    if not instructions:
        raise ValueError("Skill instructions 不能为空")

    return SkillDefinition(
        name=skill_name,
        description=description,
        instructions=instructions,
        directory=directory,
    )


def list_skills(skills_dir: Path = DEFAULT_SKILLS_DIR) -> list[SkillDefinition]:
    if not skills_dir.exists():
        return []
    skills = []
    for directory in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
        if (directory / "SKILL.md").is_file():
            skills.append(load_skill(directory.name, skills_dir=skills_dir))
    return skills

