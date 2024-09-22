package code

import (
	"math"
	"math/bits"
	"slices"
	"sort"
)

func maxSubArray(nums []int) int {
	res := nums[0]
	maxArray := make([]int, len(nums))
	for i := range nums {
		if i == 0 {
			maxArray[i] = nums[i]
			continue
		}
		if maxArray[i-1] <= 0 {
			maxArray[i] = nums[i]
		} else {
			maxArray[i] = maxArray[i-1] + nums[i]
		}
		if maxArray[i] > res {
			res = maxArray[i]
		}
	}
	return res
}

func convert(s string, numRows int) string {
	if numRows < 2 {
		return s
	}
	resArray := make([]string, numRows)
	row := 0
	flag := -1
	for _, c := range s {
		if row == 0 || row == numRows-1 {
			flag = -flag
		}
		resArray[row] = resArray[row] + string(c)
		row += flag
	}
	res := ""
	for i := range resArray {
		res = res + resArray[i]
	}
	return res
}
func combinationSum4(nums []int, target int) int {
	memo := make([]int, target+1)
	for i := range nums {
		memo[i] = -1
	}
	var dfs func(int) int
	dfs = func(i int) (res int) {
		if i == 0 {
			return 0
		}
		p := &memo[i]
		if *p != -1 {
			return *p
		}
		for _, num := range nums {
			if num <= i {
				res += dfs(i - num)
			}
		}
		*p = res
		return
	}
	return dfs(target)
}

func minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := range dp {
		dp[i] = make([]int, len(word2)+1)
	}
	for i := range dp {
		dp[i][0] = i
	}
	for j := range dp[0] {
		dp[0][j] = j
	}

	for i := 1; i < len(dp); i++ {
		for j := 0; j < len(dp[i]); j++ {
			if word1[i-1] == word2[i-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = minOfThree(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
			}
		}
	}
	return dp[len(word1)][len(word2)]
}

func minOfThree(a, b, c int) int {
	minAB := a
	if b < minAB {
		minAB = b
	}
	if c < minAB {
		minAB = c
	}
	return minAB
}

func rob(nums []int) int {
	dp := make([]int, len(nums))
	rob := nums[0]
	for i := range nums {
		if i == 0 {
			dp[i] = nums[i]
		} else if i == 1 {
			if nums[i] > nums[i-1] {
				dp[i] = nums[i]
			} else {
				dp[i] = nums[i-1]
			}
		} else {
			if dp[i-2]+nums[i] > dp[i-1] {
				dp[i] = dp[i-2] + nums[i]
			} else {
				dp[i] = dp[i-1]
			}

		}
		if dp[i] > rob {
			rob = dp[i]
		}
	}
	return rob
}

// 注意：go 代码由 chatGPT🤖 根据我的 java 代码翻译，旨在帮助不同背景的读者理解算法逻辑。
// 本代码不保证正确性，仅供参考。如有疑惑，可以参照我写的 java 代码对比查看。

func wordBreak(s string, wordDict []string) bool {
	var wordBreakDFS func(s string, wordDict []string, start int, memo map[int]bool) bool
	wordBreakDFS = func(s string, wordDict []string, start int, memo map[int]bool) bool {
		if start == len(s) {
			return true
		}
		if memo[start] {
			return false
		}
		for _, word := range wordDict {
			if start+len(word) <= len(s) && s[start:start+len(word)] == word {
				if wordBreakDFS(s, wordDict, start+len(word), memo) {
					return true
				}
			}
		}
		memo[start] = true
		return false
	}
	return wordBreakDFS(s, wordDict, 0, make(map[int]bool))
}

func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		dp[i][0] = 1
	}
	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}

func maximalSquare(matrix [][]byte) int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return 0
	}

	rows, columns := len(matrix), len(matrix[0])
	dp := make([][]int, rows)
	maxSize := 0

	for i := range dp {
		dp[i] = make([]int, columns)
		dp[i][0] = int(matrix[i][0] - '0')
		maxSize = max(maxSize, dp[i][0])
	}

	for j := range dp[0] {
		dp[0][j] = int(matrix[0][j] - '0')
		maxSize = max(maxSize, dp[0][j])
	}

	for i := 1; i < rows; i++ {
		for j := 1; j < columns; j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1])) + 1
				maxSize = max(maxSize, dp[i][j])
			}
		}
	}

	return maxSize * maxSize
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func numTrees(n int) int {
	use := make([]int, n)
	return cacluateTree(0, &use, 0, n)
}

func cacluateTree(deep int, use *[]int, size int, n int) int {
	if deep == len(*use) {
		return 1
	}
	for i := 0; i < n; i++ {
		if (*use)[i] == 1 {
			continue
		}
		(*use)[i] = 1
		size += cacluateTree(deep+1, use, size, n)
		(*use)[i] = 0
	}
	return size
}

func permuteUnique(nums []int) [][]int {
	var res [][]int
	sort.Ints(nums)
	dfs(&res, nums, 0, make([]int, len(nums)), make([]bool, len(nums)))
	return res
}

func dfs(res *[][]int, nums []int, deep int, every []int, use []bool) {
	if deep == len(nums) {
		newRes := make([]int, len(every))
		copy(newRes, every)
		*res = append(*res, newRes)
		return
	}
	for i := range nums {
		if use[i] {
			continue
		}
		if i != 0 && (nums[i] == nums[i-1] && !use[i-1]) {
			continue
		}
		use[i] = true
		every = append(every, nums[i])
		dfs(res, nums, deep+1, every, use)
		use[i] = false
		every = append(every[0 : len(every)-1])
	}
}

func sortColors(nums []int) {
	// 插入
	for i := range nums {
		if i == 0 {
			continue
		}
		var minIndex = i
		var minNum = nums[i]
		for j := i - 1; j >= 0; j-- {
			if nums[j] < minNum {
				minIndex = j
				minNum = nums[j]
			} else {
				break
			}
		}
		if minIndex == i {
			continue
		} else {
			for j := i; j >= minIndex; j-- {
				var temp = nums[j]
				nums[j] = nums[j-1]
				nums[j-1] = temp
			}
		}
	}
}

func searchMatrix(matrix [][]int, target int) bool {
	var row = 0
	var maxLen = len(matrix[0]) - 1
	for ; row <= len(matrix)-1; row++ {
		for j := 0; j < maxLen; j++ {
			if target == matrix[row][j] {
				return true
			}
			if target < matrix[row][j] && j == 0 {
				return false
			}
			if target < matrix[row][j] {
				maxLen = j
			}
		}
	}
	return false
}

type Trie struct {
	end   bool
	child [26]*Trie
}

func Constructor() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string) {
	node := this
	for _, ch := range word {
		index := ch - 'a'
		if nil == node.child[index] {
			node.child[index] = &Trie{}
		}
		node = node.child[index]
	}
	node.end = true
}

func (this *Trie) Search(word string) bool {
	return this.StartsWithPrefix(word) != nil && this.StartsWithPrefix(word).end
}

func (this *Trie) StartsWith(prefix string) bool {
	return this.StartsWithPrefix(prefix) != nil
}

func (this *Trie) StartsWithPrefix(prefix string) *Trie {
	node := this
	for _, ch := range prefix {
		index := ch - 'a'
		if node.child[index] == nil {
			return nil
		}
		node = node.child[index]
	}
	return node
}

func canFinish(numCourses int, prerequisites [][]int) bool {
	var (
		visited       = make([]int, numCourses)
		edges         = make([][]int, numCourses)
		canFinishNums = 0
	)
	for _, info := range prerequisites {
		edges[info[1]] = append(edges[info[1]], info[0])
		visited[info[0]]++
	}
	q := []int{}
	for i, num := range visited {
		if num == 0 {
			q = append(q, i)
		}
	}
	for len(q) > 0 {
		canFinishNums++
		u := q[0]
		q = q[1:]
		after := edges[u]
		for _, afterCourse := range after {
			visited[afterCourse]--
			if visited[afterCourse] == 0 {
				q = append(q, afterCourse)
			}
		}
	}
	return canFinishNums == numCourses
}

//func canFinish(numCourses int, prerequisites [][]int) bool {
//	var (
//		visited      = make([]int, numCourses)
//		edges        = make([][]int, numCourses)
//		valid        = true
//		canFinishDfs func(i int)
//	)
//	for _, info := range prerequisites {
//		edges[info[1]] = append(edges[info[1]], info[0])
//	}
//	canFinishDfs = func(i int) {
//		visited[i] = 1
//		for _, pre := range edges[i] {
//			if visited[pre] == 0 {
//				canFinishDfs(pre)
//				if !valid {
//					return
//				}
//			} else if visited[pre] == 1 {
//				valid = false
//				return
//			}
//		}
//		visited[i] = 2
//	}
//	for i := 0; i < numCourses; i++ {
//		if visited[i] == 0 {
//			canFinishDfs(i)
//		}
//		if !valid {
//			return false
//		}
//	}
//	return valid
//}

func minPathSum(grid [][]int) int {
	minPath := make([][]int, len(grid))
	for i := range grid {
		minPath[i] = make([]int, len(grid[i]))
	}
	minPath[0][0] = grid[0][0]
	for j := 1; j < len(grid[0]); j++ {
		minPath[0][j] = minPath[0][j-1] + grid[0][j]
	}
	// 初始化 minPath 的第一列
	for i := 1; i < len(grid); i++ {
		minPath[i][0] = minPath[i-1][0] + grid[i][0]
	}
	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[0]); j++ {
			if minPath[i][j-1] < minPath[i-1][j] {
				minPath[i][j] = minPath[i][j-1] + grid[i][j]
			} else {
				minPath[i][j] = minPath[i-1][j] + grid[i][j]
			}
		}
	}
	return minPath[len(grid)-1][len(grid[0])-1]
}

func maxArea(height []int) int {
	left := height[0]
	right := height[len(height)-1]
	leftIndex := 0
	rightIndex := len(height) - 1
	maxArea := 0
	for leftIndex < rightIndex {
		if height[leftIndex] > left {
			left = height[leftIndex]
		}
		if height[rightIndex] > right {
			right = height[rightIndex]
		}
		tempMaxArea := 0
		if left > right {
			tempMaxArea = (rightIndex - leftIndex) * right
		} else {
			tempMaxArea = (rightIndex - leftIndex) * left
		}
		if tempMaxArea > maxArea {
			maxArea = tempMaxArea
		}
		leftIndex++
		rightIndex--
	}
	return maxArea
}

func maxProfit(prices []int) int {
	// 0 有一支 1 没有冷冻 2 没有
	dp := make([][]int, len(prices))
	for i := 0; i < len(prices); i++ {
		dp[i] = make([]int, 3)
	}
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	dp[0][2] = 0
	res := 0
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][2]-prices[i], dp[i-1][0])
		dp[i][1] = dp[i-1][0] + prices[i]
		dp[i][2] = max(dp[i-1][2], dp[i-1][1])
		res = max(dp[i][0], res)
		res = max(dp[i][1], res)
		res = max(dp[i][2], res)
	}
	return res
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}
	return left
}

func productExceptSelf(nums []int) []int {
	temp := 1
	answ := make([]int, len(nums))
	answ[0] = 1
	for i := 1; i < len(nums); i++ {
		answ[i] = answ[i-1] * nums[i-1]
	}
	for i := len(nums) - 2; i >= 0; i-- {
		temp *= nums[i+1]
		answ[i] *= temp
	}
	return answ
}

func numSquares(n int) int {
	dp := make([]int, n+1)
	dp[0] = 0
	for i := 1; i <= n; i++ {
		dp[i] = i
		for j := 1; i-j*j >= 0; j++ {
			if dp[i] > dp[i-j*j]+1 {
				dp[i] = dp[i-j*j] + 1
			}
		}
	}
	return dp[n]
}

func findDuplicate(nums []int) int {
	lenArray := len(nums) - 1
	left := 0
	right := 1
	for {
		if nums[left%lenArray] == nums[right%lenArray] {
			return nums[left%lenArray]
		} else {
			left++
			right += 2
		}
	}
}

func groupAnagrams(strs []string) [][]string {
	groupAnagramsMap := make(map[string][]string)
	for _, str := range strs {
		temp := []byte(str)
		slices.Sort(temp)
		sortStr := string(temp)
		groupAnagramsMap[sortStr] = append(groupAnagramsMap[sortStr], str)
	}
	res := make([][]string, len(groupAnagramsMap))
	for _, strings := range groupAnagramsMap {
		res = append(res, strings)
	}
	return res
}

func removeDuplicates(nums []int) int {
	lastPos := 0
	curNum := nums[0]
	curTime := 0
	for _, num := range nums {
		curTime++
		if curTime > 2 && curNum == num {
			continue
		}
		if curNum != num {
			curTime = 1
			curNum = num
		}
		nums[lastPos] = num
		lastPos++
	}
	return lastPos
}

func maxProfit2(prices []int) int {
	dp := make([][]int, len(prices))
	for i := 0; i < len(prices); i++ {
		dp[i] = make([]int, 2)
	}
	dp[0][0] = 0
	dp[0][1] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		dp[i][1] = max(dp[i-1][0]-prices[i], dp[i-1][1])
	}
	return dp[len(prices)-1][0]
}

func maxProfit3(prices []int) int {
	dp := make([][]int, len(prices))
	for i := 0; i < len(prices); i++ {
		dp[i] = make([]int, 4)
	}
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], -prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
		dp[i][2] = max(dp[i-1][1]-prices[i], dp[i-1][2])
		dp[i][3] = max(dp[i-1][2]+prices[i], dp[i-1][3])
	}
	return dp[len(prices)-1][3]
}

func accountsMerge(accounts [][]string) [][]string {
	indexMapByEmail := make(map[string][]int)
	for index, account := range accounts {
		for i := 1; i < len(account); i++ {
			indexMapByEmail[account[i]] = append(indexMapByEmail[account[i]], index)
		}
	}
	emailSet := map[string]struct{}{}
	vis := make([]bool, len(accounts))
	dfsAccount := func(i int) {}
	dfsAccount = func(i int) {
		vis[i] = true
		for _, email := range accounts[i][1:] {
			if _, has := emailSet[email]; has {
				continue
			}
			emailSet[email] = struct{}{}
			for _, account := range indexMapByEmail[email] {
				dfsAccount(account)
			}
		}
	}
	res := make([][]string, 0)
	for i, vi := range vis {
		if vi {
			continue
		}
		clear(emailSet)
		dfsAccount(i)
		account := make([]string, 0)
		account = append(account, accounts[i][0])
		for s := range emailSet {
			account = append(account, s)
		}
		sort.Strings(account[1:])
		res = append(res, account)
	}
	return res
}
func climbStairs(n int) int {
	if n == 1 {
		return 1
	}
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 1
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
func minCostClimbingStairs(cost []int) int {
	dp := make([]int, len(cost)+1)
	dp[0] = 0
	dp[1] = 0
	for i := 2; i <= len(cost); i++ {
		dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
	}
	return dp[len(cost)]
}

func maxHeightOfTriangle(red int, blue int) int {
	count := 0
	dfsColor := func(colorIndex int, row int) {}
	dfsColor = func(colorIndex int, row int) {
		if colorIndex == 0 {
			red = red - (2*row - row)
			if red >= 0 {
				count++
				dfsColor(1, row+1)
			}
		} else {
			blue = blue - (2*row - row)
			if blue >= 0 {
				count++
				dfsColor(0, row+1)
			}
		}
		if colorIndex == 0 {
			red = red + (2*row - row)
		} else {
			blue = blue + (2*row - row)
		}
	}
	dfsColor(0, 1)
	countRed := count
	count = 0
	dfsColor(1, 1)
	countBlue := count
	return max(countRed, countBlue)
}
func minRectanglesToCoverPoints(points [][]int, w int) int {

	sort.Slice(points, func(i, j int) bool {
		return points[i][0] < points[j][0]
	})
	res := 0
	bound := -1
	for _, elm := range points {
		if elm[0] > bound {
			bound = elm[0] + w
			res++
		}
	}
	return res
}

func numberOfRightTriangles(grid [][]int) int64 {
	rowNum := make([]int64, len(grid))
	colNum := make([]int64, len(grid[0]))
	for i, _ := range grid {
		for j, _ := range grid[i] {
			if grid[i][j] == 1 {
				rowNum[i]++
				colNum[j]++
			}
		}
	}
	res := int64(0)
	for i, _ := range grid {
		for j, _ := range grid[i] {
			if grid[i][j] == 1 {
				res += (rowNum[i] - 1) * (colNum[j] - 1)
			}
		}
	}
	return res
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	res := new(ListNode)
	temp := res
	carry := 0

	for l1 != nil || l2 != nil || carry > 0 {
		sum := carry
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}

		carry = sum / 10
		temp.Val = sum % 10

		// Only create a new node if there are more digits to process
		if l1 != nil || l2 != nil || carry > 0 {
			temp.Next = new(ListNode)
			temp = temp.Next
		}
	}

	return res
}

func isArraySpecial(nums []int, queries [][]int) []bool {
	dp := make([]int, len(nums))
	res := make([]bool, len(queries))
	// 填充 dp 表格
	for i := 1; i < len(nums); i++ {
		dp[i] = dp[i-1]
		if nums[i-1]%2 == nums[i]%2 {
			dp[i]++
		}
	}
	for i, query := range queries {
		res[i] = dp[query[0]] == dp[query[1]]
	}
	return res
}

func maxScore(grid [][]int) int {
	lowScoreGrid := make([][]int, len(grid)+1)
	res := math.MinInt
	lowScoreGrid[0] = make([]int, len(grid[0])+1)
	for i := range lowScoreGrid[0] {
		lowScoreGrid[0][i] = math.MaxInt
	}
	for i, nums := range grid {
		lowScoreGrid[i+1] = make([]int, len(nums)+1)
		lowScoreGrid[i+1][0] = math.MaxInt
		for j, num := range nums {
			lowScoreGrid[i+1][j+1] = min(lowScoreGrid[i][j+1], lowScoreGrid[i+1][j])
			nowMax := num - lowScoreGrid[i+1][j+1]
			lowScoreGrid[i+1][j+1] = min(lowScoreGrid[i+1][j+1], num)
			res = max(res, nowMax)
		}
	}
	return res
}

func findMaximumNumber(k int64, x int) int64 {
	ans := sort.Search(int(k+1)<<x, func(num int) bool {
		num++
		n := bits.Len(uint(num))
		memo := make([][]int, n)
		for i := range memo {
			memo[i] = make([]int, n+1)
			for j := range memo[i] {
				memo[i][j] = -1
			}
		}
		var dfs func(int, int, bool) int
		dfs = func(i, cnt1 int, limitHigh bool) (res int) {
			if i < 0 {
				return cnt1
			}
			if !limitHigh {
				p := &memo[i][cnt1]
				if *p >= 0 {
					return *p
				}
				defer func() { *p = res }()
			}
			up := 1
			if limitHigh {
				up = num >> i & 1
			}
			for d := 0; d <= up; d++ {
				c := cnt1
				if d == 1 && (i+1)%x == 0 {
					c++
				}
				res += dfs(i-1, c, limitHigh && d == up)
			}
			return
		}
		return dfs(n-1, 0, true) > int(k)
	})
	return int64(ans)
}
func countWays(nums []int) int {
	sort.Ints(nums)
	ans := 0
	if nums[0] > 0 {
		ans++
	}
	for i := 1; i < len(nums); i++ {
		if nums[i] > i && nums[i-1] < i {
			ans++
		}
	}
	if nums[len(nums)-1] < len(nums) {
		ans++
	}
	return ans
}
func maxStrength(nums []int) int64 {
	mn, mx := nums[0], nums[0]
	for _, x := range nums[1:] {
		mn, mx = minFour(mn, x, mn*x, mx*x),
			maxFour(mx, x, mn*x, mx*x)
	}
	return int64(mx)
}

// 实现 min 函数
func minFour(a, b, c, d int) int {
	min := a
	if b < min {
		min = b
	}
	if c < min {
		min = c
	}
	if d < min {
		min = d
	}
	return min
}

// 实现 max 函数
func maxFour(a, b, c, d int) int {
	max := a
	if b > max {
		max = b
	}
	if c > max {
		max = c
	}
	if d > max {
		max = d
	}
	return max
}

func maxThree(num1 int, num2 int, num3 int) int {
	maxNum := num1
	if maxNum < num2 {
		maxNum = num2
	}
	if maxNum < num3 {
		maxNum = num3
	}
	return maxNum
}

func minThree(num1 int, num2 int, num3 int) int {
	minNum := num1
	if minNum > num2 {
		minNum = num2
	}
	if minNum > num3 {
		minNum = num3
	}
	return minNum
}
