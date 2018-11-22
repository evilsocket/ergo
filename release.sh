#!/bin/bash
# nothing to see here, just a utility i use to create new releases ^_^

VERSION_FILE=ergo/version.py
CURRENT_VERSION=$(cat ${VERSION_FILE} | grep "__version__" | cut -d"'" -f2)

echo -n "Current version is $CURRENT_VERSION, select new version: "
read NEW_VERSION
echo "Creating version $NEW_VERSION ...\n"

echo "Patching ${VERSION_FILE} ..."
sed -i "s/$CURRENT_VERSION/$NEW_VERSION/g" "${VERSION_FILE}"
git add "${VERSION_FILE}"

git commit -m "Releasing v$NEW_VERSION"
git push

git tag -a v$NEW_VERSION -m "Release v$NEW_VERSION"
git push origin v$NEW_VERSION

echo
echo "All done, v$NEW_VERSION released ^_^"
